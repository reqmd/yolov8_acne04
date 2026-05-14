from src.data.dataset import AcneDataset, collate_fn
from src.model.model import YoloModel
from src.utils.data_utils import make_anchors, decode_predictions, tal_matcher, xywh2xyxy
from src.model.losses import LossFunction
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def post_process(pred_boxes, pred_cls, conf_threshold=0.25, iou_threshold=0.45):
    """
    pred_boxes: (B, N, 4) — боксы в пикселях x1y1x2y2
    pred_cls:   (B, 1, N) — логиты confidence
    
    Возвращает список словарей для каждого изображения в батче
    """
    B = pred_boxes.shape[0]
    results = []

    conf = pred_cls[:, 0, :].sigmoid()  # (B, N) — переводим в вероятности

    for b in range(B):
        scores = conf[b]                # (N,)
        boxes  = pred_boxes[b]          # (N, 4)

        # Фильтруем по порогу уверенности
        mask   = scores > conf_threshold
        scores = scores[mask]
        boxes  = boxes[mask]

        if len(scores) == 0:
            results.append({'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0)})
            continue

        # NMS
        keep   = nms(boxes, scores, iou_threshold)
        results.append({
            'boxes':  boxes[keep],   # (K, 4)
            'scores': scores[keep]   # (K,)
        })

    return results

def evaluate(model, val_loader, anchor_points, stride_tensor,
             conf_threshold=0.25, iou_threshold=0.45, img_size=1280):
    
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox')

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Evaluating'):
            images  = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            pred_boxes, pred_cls, pred_dist = decode_predictions(
                outputs, anchor_points, stride_tensor
            )

            # NMS
            results = post_process(pred_boxes, pred_cls,
                                   conf_threshold, iou_threshold)

            B = images.shape[0]

            # Формируем предсказания для metric
            preds = []
            for b in range(B):
                preds.append({
                    'boxes':  results[b]['boxes'].cpu(),
                    'scores': results[b]['scores'].cpu(),
                    'labels': torch.zeros(
                        len(results[b]['scores']), dtype=torch.long
                    )  # класс 0 для всех
                })

            # Формируем GT для metric
            gts = []
            for b in range(B):
                obj_mask = targets[:, 0] == b
                gt_boxes = targets[obj_mask, 2:]  # xywh нормализованный

                # Конвертируем в пиксельный x1y1x2y2
                if len(gt_boxes) > 0:
                    gt_boxes = xywh2xyxy(gt_boxes, img_size).cpu()
                    gt_labels = torch.zeros(len(gt_boxes), dtype=torch.long)
                else:
                    gt_boxes  = torch.zeros((0, 4))
                    gt_labels = torch.zeros(0, dtype=torch.long)

                gts.append({
                    'boxes':  gt_boxes,
                    'labels': gt_labels
                })

            metric.update(preds, gts)

    result = metric.compute()
    return result

if __name__ == '__main__':
    #show_model_info(mod='s')
    train_data = AcneDataset('data/Annotations/train.csv', train=True)
    val_data = AcneDataset('data/Annotations/val.csv', train=False)
    epochs = 25
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

    device = 'cuda'
    model = YoloModel(mod='n').to(device)
    #model = torch.compile(model)
    scaler = GradScaler()
    anchor_points, stride_tensor = make_anchors(img_size=1280)
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)
    criterion = LossFunction().to(device)
    optim = torch.optim.AdamW(params=model.parameters(), lr=0.002, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)

    # История лоссов раздельно для train и val
    history = {
        'train': {'total': [], 'ciou': [], 'dfl': [], 'cls': []},
        'val':   {'total': [], 'ciou': [], 'dfl': [], 'cls': []}
    }

    
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
            scaler.unscale_(optim) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optim)
            scaler.update()

            train_losses['total'].append(loss.item())
            train_losses['ciou'].append(loss_dict['ciou'])
            train_losses['dfl'].append(loss_dict['dfl'])
            train_losses['cls'].append(loss_dict['cls'])

        # ── Validation ────────────────────────────────────────────
        # model.eval()
        # val_losses = {'total': [], 'ciou': [], 'dfl': [], 'cls': []}
        
        # val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ')
        # with torch.no_grad():
        #     for images, targets in val_bar:
        #         images  = images.to(device)
        #         targets = targets.to(device)

        #         outputs = model(images)
        #         pred_boxes, pred_cls, pred_dist = decode_predictions(
        #             outputs, anchor_points, stride_tensor
        #         )
        #         positive_mask, matched_boxes, matched_scores = tal_matcher(
        #             pred_boxes, pred_cls, targets,
        #             anchor_points, stride_tensor, img_size=1280
        #         )
        #         loss, loss_dict = criterion(
        #             pred_boxes, pred_dist, pred_cls,
        #             positive_mask, matched_boxes, matched_scores,
        #             anchor_points, stride_tensor
        #         )

        #         val_losses['total'].append(loss.item())
        #         val_losses['ciou'].append(loss_dict['ciou'])
        #         val_losses['dfl'].append(loss_dict['dfl'])
        #         val_losses['cls'].append(loss_dict['cls'])

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
            map_result = evaluate(
                model, val_loader,
                anchor_points, stride_tensor
            )
            print(f'mAP@50:    {map_result["map_50"]:.4f}')
            print(f'mAP@50-95: {map_result["map"]:.4f}')

    # ── Сохранение модели ──────────────────────────────────────────
    torch.save({
        'epoch':       epochs,
        'model':       model.state_dict(),
        'optimizer':   optim.state_dict(),
        'history':     history,
    }, 'checkpoint.pth')