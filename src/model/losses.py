from src.utils.data_utils import bbox2dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss as ciou_loss

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
        """
        pred_boxes:     (B, N, 4)  декодированные боксы в пикселях
        pred_dist:      (B, 64, N) сырые логиты DFL
        pred_cls:       (B, 1, N)  сырые логиты objectness
        positive_mask:  (B, N)     маска позитивных ячеек
        matched_boxes:  (B, N, 4)  GT боксы в пикселях
        matched_scores: (B, N)     IoU качество совпадения
        anchor_points:  (N, 2)
        stride_tensor:  (N, 1)
        """
        dtype = pred_cls.dtype

        matched_boxes  = matched_boxes.to(dtype)
        matched_scores = matched_scores.to(dtype)
        
        n_pos = positive_mask.sum().clamp(min=1)  # число позитивных ячеек

        # ── CIoU Loss — только позитивные ячейки ─────────────────
        pred_boxes_pos   = pred_boxes[positive_mask]    # (M, 4)
        matched_boxes_pos = matched_boxes[positive_mask] # (M, 4)

        if pred_boxes_pos.shape[0] > 0:
            loss_ciou = ciou_loss(
                pred_boxes_pos,
                matched_boxes_pos,
                reduction='mean'
            )
        else:
            loss_ciou = torch.tensor(0.0, device=pred_boxes.device)

        # ── DFL Loss — только позитивные ячейки ──────────────────
        # Конвертируем GT боксы в расстояния для DFL
        target_dist = bbox2dist(
            matched_boxes, anchor_points, stride_tensor
        )  # (B, N, 4)

        # pred_dist: (B, 64, N) → (B, N, 64)
        pred_dist_pos   = pred_dist.permute(0, 2, 1)[positive_mask]   # (M, 64)
        target_dist_pos = target_dist[positive_mask]                   # (M, 4)

        if pred_dist_pos.shape[0] > 0:
            loss_dfl = dfl_loss(pred_dist_pos, target_dist_pos)
        else:
            loss_dfl = torch.tensor(0.0, device=pred_boxes.device)

        # ── BCE Loss — все ячейки ─────────────────────────────────
        # Цель: позитивные=1, негативные=0
        # Взвешиваем на matched_scores чтобы лучшие совпадения важнее
        cls_target = torch.zeros_like(pred_cls[:, 0, :])  # (B, N)
        cls_target[positive_mask] = matched_scores[positive_mask]
        with torch.amp.autocast('cuda', enabled=False):
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls[:, 0, :].float(),   # (B, N)
                cls_target.float(),          # (B, N)
                reduction='mean'
            ) / n_pos

        # ── Итоговый loss ─────────────────────────────────────────
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