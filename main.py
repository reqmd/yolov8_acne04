from src.data.dataset import AcneDataset, collate_fn
from src.model.blocks import ConvBlock, BottleNeckBlock, SPPFBlock, C2fBlock
from src.model.test_losses import test_ciou_loss, test_dfl_loss
from src.utils.show_model_info import show_model_info
from src.utils.transforms import return_transforms
from src.model.model import YoloModel
from src.utils.data_utils import make_anchors, decode_predictions, tal_matcher
from src.model.losses import LossFunction

import torch
import numpy as nn
from torch.utils.data import DataLoader
import time

#show_model_info(mod='s')
train_data = AcneDataset('data/Annotations/train.csv', train=True)
val_data = AcneDataset('data/Annotations/val.csv', train=False)

train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

device = 'cuda'
tens = torch.zeros((1, 3, 1280, 1280))
model = YoloModel(mod = 'n').to(device)

anchor_points, stride_tensor = make_anchors(img_size=1280)
anchor_points = anchor_points.to(device)
stride_tensor = stride_tensor.to(device)
criterion = LossFunction().to(device)
optim = torch.optim.Adam(params=model.parameters(), lr = 0.001)

total_loss = []
cls_loss = []
dfl_loss = []
ciou_loss = []

epochs = 25
for epoch in range(epochs):
    start_time = time.time()
    for images, targets in train_loader:
        #print(images.shape)
        #print(targets.shape)
        images  = images.to(device)
        targets = targets.to(device)

        optim.zero_grad()

        # 1. Прямой проход
        outputs = model(images)
        #print(outputs[0][0].shape, outputs[1][0].shape)

        # 2. Декодируем предсказания
        pred_boxes, pred_cls, pred_dist = decode_predictions(
            outputs, anchor_points, stride_tensor
        )

        # 3. Matcher
        positive_mask, matched_boxes, matched_scores = tal_matcher(
            pred_boxes, pred_cls, targets,
            anchor_points, stride_tensor, img_size=1280
        )

        loss, loss_dict = criterion(
            pred_boxes, pred_dist, pred_cls,
            positive_mask, matched_boxes, matched_scores,
            anchor_points, stride_tensor
        )

        # print(f"total={loss.item():.4f} | "
        #     f"ciou={loss_dict['ciou']:.4f} | "
        #     f"dfl={loss_dict['dfl']:.4f} | "
        #     f"cls={loss_dict['cls']:.4f}")

        loss.backward()
        optim.step()
        cls_loss.append(loss_dict['cls'])
        dfl_loss.append(loss_dict['dfl'])
        ciou_loss.append(loss_dict['ciou'])
        total_loss.append(loss.item())
    end_time = time.time()
    print('//////////////////////////////////////////////////////////')
    print(f'Epoch {epoch+1} / {epochs}')
    print(f'CIOU Loss {nn.mean(ciou_loss):.4f}')
    print(f'DFL Loss {nn.mean(dfl_loss):.4f}')
    print(f'CLS Loss {nn.mean(cls_loss):.4f}')
    print(f'Total Loss {nn.mean(total_loss):.4f}')
    sec = (end_time - start_time) // 60
    min = (end_time - start_time) % 60
    print(f'Время выполнения: {min} минут {sec}')
    print('//////////////////////////////////////////////////////////')
        