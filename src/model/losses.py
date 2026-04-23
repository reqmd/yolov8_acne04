import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss as ciou_loss

def dfl_loss(pred, target, reg_max=16):
    N = pred.shape[0]
    pred = pred.view(N * 4, reg_max)
    target = target.view(-1)

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
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = 1.5 * dfl_loss(pred, target) + 7.5 * ciou_loss(pred, target)
        return loss.mean()