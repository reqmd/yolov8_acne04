from src.data.dataset import AcneDataset, collate_fn
from src.model.model import YoloModel
from src.utils.data_utils import make_anchors, decode_predictions, tal_matcher
from src.model.losses import LossFunction
from src.utils.train_utils import evaluate, load_params, plot_losses

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm
from datetime import datetime

#show_model_info(mod='s')
YAML_ROOT = 'config/preprocessing.yaml'
epochs, batch_size, lr, weight_decay, num_workers, pin_memory, persistent_workers, mod, device = load_params(YAML_ROOT)

train_data = AcneDataset('data/Annotations/train.csv', train=True)
val_data = AcneDataset('data/Annotations/val.csv', train=False)

train_loader = DataLoader(dataset=train_data, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn, 
                          num_workers=num_workers, 
                          pin_memory=pin_memory, 
                          persistent_workers=persistent_workers)

val_loader = DataLoader(dataset=val_data, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=collate_fn, 
                        num_workers=num_workers, 
                        pin_memory=pin_memory, 
                        persistent_workers=persistent_workers)

model = YoloModel(mod=mod).to(device)
scaler = GradScaler()
anchor_points, stride_tensor = make_anchors(img_size=1280)
anchor_points = anchor_points.to(device)
stride_tensor = stride_tensor.to(device)
criterion = LossFunction(lambda_ciou=7.5, lambda_dfl=1.5, lambda_cls=1.5).to(device)

optim = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-6, T_max=epochs)

# История лоссов раздельно для train и val
history = {
    'train': {'total': [], 'ciou': [], 'dfl': [], 'cls': []},
    'val':   {'total': [], 'ciou': [], 'dfl': [], 'cls': []}
}

accumulation_steps = 4

for epoch in range(epochs):
    start_time = time.time()

    # ── Train ─────────────────────────────────────────────────
    model.train()
    train_losses = {'total': [], 'ciou': [], 'dfl': [], 'cls': []}

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
    for i, (images, targets) in enumerate(train_bar):
        images  = images.to(device)
        targets = targets.to(device)

        with autocast(device_type=device):
            outputs = model(images)
            pred_boxes, pred_cls, pred_dist = decode_predictions(
                outputs, anchor_points, stride_tensor
            )
            positive_mask, matched_boxes, matched_scores = tal_matcher(
                pred_boxes, pred_cls, targets,
                anchor_points, stride_tensor, img_size=1280
            )
            loss, loss_dict = criterion(
                pred_boxes, pred_dist, pred_cls,
                positive_mask, matched_boxes, matched_scores,
                anchor_points, stride_tensor
            )
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        train_losses['total'].append(loss.item())
        train_losses['ciou'].append(loss_dict['ciou'])
        train_losses['dfl'].append(loss_dict['dfl'])
        train_losses['cls'].append(loss_dict['cls'])

    # ── Логирование ───────────────────────────────────────────
    for key in ['total', 'ciou', 'dfl', 'cls']:
        history['train'][key].append(torch.tensor(train_losses[key]).mean().item())
        #history['val'][key].append(torch.tensor(val_losses[key]).mean().item())

    end_time = time.time()
    mins = int((end_time - start_time) // 60)
    secs = int((end_time - start_time) % 60)

    print('=' * 60)
    print(f'Epoch {epoch+1}/{epochs}  |  {mins}м {secs}с')
    print(f'{"":10} {"Train":>10} ') # {"Val":>10}
    print(f'{"Total":10} {history["train"]["total"][-1]:>10.4f} ') # {history["val"]["total"][-1]:>10.4f}
    print(f'{"CIoU":10} {history["train"]["ciou"][-1]:>10.4f} ') # {history["val"]["ciou"][-1]:>10.4f}
    print(f'{"DFL":10} {history["train"]["dfl"][-1]:>10.4f} ') # {history["val"]["dfl"][-1]:>10.4f}
    print(f'{"CLS":10} {history["train"]["cls"][-1]:>10.4f} ') # {history["val"]["cls"][-1]:>10.4f}
    print('=' * 60)
    scheduler.step()

    # Логируем текущий lr
    current_lr = scheduler.get_last_lr()[0]
    print(f'LR: {current_lr:.8f}')

    if (epoch + 1) % 3 == 0:
        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_losses = {'total': [], 'ciou': [], 'dfl': [], 'cls': []}
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ')
        with torch.no_grad():
            for images, targets in val_bar:
                images  = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                pred_boxes, pred_cls, pred_dist = decode_predictions(
                    outputs, anchor_points, stride_tensor
                )
                positive_mask, matched_boxes, matched_scores = tal_matcher(
                    pred_boxes, pred_cls, targets,
                    anchor_points, stride_tensor, img_size=1280
                )
                loss, loss_dict = criterion(
                    pred_boxes, pred_dist, pred_cls,
                    positive_mask, matched_boxes, matched_scores,
                    anchor_points, stride_tensor
                )

                val_losses['total'].append(loss.item())
                val_losses['ciou'].append(loss_dict['ciou'])
                val_losses['dfl'].append(loss_dict['dfl'])
                val_losses['cls'].append(loss_dict['cls'])  
        map_result = evaluate(
            model, val_loader,
            anchor_points, stride_tensor
        )
        print(f'mAP@50:    {map_result["map_50"]:.4f}')
        print(f'mAP@50-95: {map_result["map"]:.4f}')
plot_losses(history=history)
# ── Сохранение модели ──────────────────────────────────────────
current = datetime.now().strftime("%d/%m/%Y-%H:%M")
model_name = f'Yolov8{mod}_{current}_{epoch}.pth'
print('End train')
print(f'Save model as {model_name}')
torch.save({
    'epoch':       epochs,
    'model':       model.state_dict(),
    'optimizer':   optim.state_dict(),
    'history':     history,
}, model_name)