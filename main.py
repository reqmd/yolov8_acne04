from src.data.dataset import AcneDataset, collate_fn
from src.model.model import YoloModel
from src.utils.data_utils import make_anchors, decode_predictions, tal_matcher
from src.model.losses import LossFunction

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm


if __name__ == '__main__':
    #show_model_info(mod='s')
    train_data = AcneDataset('data/Annotations/train.csv', train=True)
    val_data = AcneDataset('data/Annotations/val.csv', train=False)

    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

    device = 'cuda'
    model = YoloModel(mod='m').to(device)
    #model = torch.compile(model)
    scaler = GradScaler()
    anchor_points, stride_tensor = make_anchors(img_size=1280)
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)
    criterion = LossFunction().to(device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.0005)

    # История лоссов раздельно для train и val
    history = {
        'train': {'total': [], 'ciou': [], 'dfl': [], 'cls': []},
        'val':   {'total': [], 'ciou': [], 'dfl': [], 'cls': []}
    }

    epochs = 25
    for epoch in range(epochs):
        start_time = time.time()

        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_losses = {'total': [], 'ciou': [], 'dfl': [], 'cls': []}

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, targets in train_bar:
            images  = images.to(device)
            targets = targets.to(device)

            optim.zero_grad()

            with autocast(device_type="cuda"):
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

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            train_losses['total'].append(loss.item())
            train_losses['ciou'].append(loss_dict['ciou'])
            train_losses['dfl'].append(loss_dict['dfl'])
            train_losses['cls'].append(loss_dict['cls'])

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

        # ── Логирование ───────────────────────────────────────────
        for key in ['total', 'ciou', 'dfl', 'cls']:
            history['train'][key].append(torch.tensor(train_losses[key]).mean().item())
            history['val'][key].append(torch.tensor(val_losses[key]).mean().item())

        end_time = time.time()
        mins = int((end_time - start_time) // 60)
        secs = int((end_time - start_time) % 60)

        print('=' * 60)
        print(f'Epoch {epoch+1}/{epochs}  |  {mins}м {secs}с')
        print(f'{"":10} {"Train":>10} {"Val":>10}')
        print(f'{"Total":10} {history["train"]["total"][-1]:>10.4f} {history["val"]["total"][-1]:>10.4f}')
        print(f'{"CIoU":10} {history["train"]["ciou"][-1]:>10.4f} {history["val"]["ciou"][-1]:>10.4f}')
        print(f'{"DFL":10} {history["train"]["dfl"][-1]:>10.4f} {history["val"]["dfl"][-1]:>10.4f}')
        print(f'{"CLS":10} {history["train"]["cls"][-1]:>10.4f} {history["val"]["cls"][-1]:>10.4f}')
        print('=' * 60)

    # ── Сохранение модели ──────────────────────────────────────────
    torch.save({
        'epoch':       epochs,
        'model':       model.state_dict(),
        'optimizer':   optim.state_dict(),
        'history':     history,
    }, 'checkpoint.pth')