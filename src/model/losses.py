import torch
import torch.nn as nn
import torch.nn.functional as F

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
