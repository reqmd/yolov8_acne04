from src.utils.data_utils import bbox2dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss as ciou_loss

if torch.cuda.is_available(): 
    device = 'cuda'
else:
    device = 'cpu'

def dfl_loss(pred, target, reg_max=16):
    """
    pred:   (M, 64) — сырые логиты позитивных ячеек
    target: (M, 4)  — расстояния ltrb в бинах
    """
    M = pred.shape[0]
    pred   = pred.view(M * 4, reg_max)    # (M*4, 16)
    target = target.reshape(-1)           # (M*4,)

    t_left  = target.long().clamp(0, reg_max - 2)
    t_right = (t_left + 1).clamp(0, reg_max - 1)
    w_right = target - t_left.float()
    w_left  = 1.0 - w_right

    loss = (
        F.cross_entropy(pred, t_left,  reduction='none') * w_left +
        F.cross_entropy(pred, t_right, reduction='none') * w_right
    )
    return loss.mean()


class LossFunction(nn.Module):
    def __init__(self, lambda_ciou=7.5, lambda_dfl=1.5, lambda_cls=0.5):
        super().__init__()
        self.lambda_ciou = lambda_ciou
        self.lambda_dfl  = lambda_dfl
        self.lambda_cls  = lambda_cls

    def forward(self, pred_boxes, pred_dist, pred_cls,
            positive_mask, matched_boxes, matched_scores,
            anchor_points, stride_tensor):

        n_pos = positive_mask.sum().clamp(min=1)
        device = pred_boxes.device

        # ── CIoU Loss ─────────────────────────────────────────
        if n_pos > 0 and positive_mask.any():
            pred_boxes_pos    = pred_boxes[positive_mask]
            matched_boxes_pos = matched_boxes[positive_mask]
            loss_ciou = ciou_loss(pred_boxes_pos, matched_boxes_pos, reduction='mean')
        else:
            loss_ciou = torch.tensor(0.0, device=device)

        # ── DFL Loss ──────────────────────────────────────────
        if n_pos > 0 and positive_mask.any():
            target_dist = bbox2dist(matched_boxes, anchor_points, stride_tensor)
            pred_dist_pos   = pred_dist.permute(0, 2, 1)[positive_mask]
            target_dist_pos = target_dist[positive_mask]
            loss_dfl = dfl_loss(pred_dist_pos, target_dist_pos)
        else:
            loss_dfl = torch.tensor(0.0, device=device)

        # ── CLS Loss ──────────────────────────────────────────
        with torch.amp.autocast(device_type='cuda'):
            cls_target = torch.zeros_like(pred_cls[:, 0, :], dtype=torch.float32)
            if positive_mask.any():
                cls_target[positive_mask] = matched_scores[positive_mask].float()
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls[:, 0, :].float(),
                cls_target,
                reduction='mean'
            )

        # ── Проверка на NaN ──────────────────────────────────
        if torch.isnan(loss_cls):
            loss_cls = torch.tensor(0.0, device=device)

        total_loss = (
            self.lambda_ciou * loss_ciou +
            self.lambda_dfl  * loss_dfl  +
            self.lambda_cls  * loss_cls
        )

        return total_loss, {
            'ciou': loss_ciou.item(),
            'dfl':  loss_dfl.item(),
            'cls':  loss_cls.item()
        }