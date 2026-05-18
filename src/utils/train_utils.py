import torch
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from src.utils.data_utils import decode_predictions, xywh2xyxy
import yaml
import matplotlib.pyplot as plt

if torch.cuda.is_available(): 
    device = 'cuda'
else:
    device = 'cpu'

def post_process(pred_boxes, pred_cls, conf_threshold=0.25, iou_threshold=0.45):
    """
    pred_boxes: (B, N, 4) — боксы в пикселях x1y1x2y2
    pred_cls:   (B, 1, N) — логиты confidence
    
    Возвращает список словарей для каждого изображения в батче
    """
    B = pred_boxes.shape[0]
    results = []

    conf = pred_cls[:, 0, :].sigmoid()

    for b in range(B):
        scores = conf[b]    
        boxes  = pred_boxes[b]    

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
            'boxes':  boxes[keep],
            'scores': scores[keep]
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

def load_params(yaml_root):
    with open(yaml_root, 'r') as file:
        data = yaml.safe_load(file)
    epochs = data['epochs']
    batch_size = data['batch_size']
    lr = data['lr']
    weight_decay = data['weight_decay']
    num_workers = data['num_workers']
    pin_memory = data['pin_memory']
    persistent_workers = data['persistent_workers']
    mod = data['mod']
    device = data['device']
    return epochs, batch_size, lr, weight_decay, num_workers, pin_memory, persistent_workers, mod, device

def plot_losses(history, save_path='losses.png'):
    epochs = range(1, len(history['train']['total']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Total Loss
    axes[0].plot(epochs, history['train']['total'], label='Train')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # CIoU Loss
    axes[1].plot(epochs, history['train']['ciou'], label='Train')
    axes[1].set_title('CIoU Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    # DFL Loss
    axes[2].plot(epochs, history['train']['dfl'], label='Train')
    axes[2].set_title('DFL Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()