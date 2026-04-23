from src.data.dataset import AcneDataset
from src.model.blocks import ConvBlock, BottleNeckBlock, SPPFBlock, C2fBlock
from src.model.test_losses import test_ciou_loss, test_dfl_loss
from src.utils.show_model_info import show_model_info
from src.utils.transforms import return_transforms
from src.model.model import YoloModel

import torch
import numpy as nn

#show_model_info(mod='s')
transform = return_transforms()
data = AcneDataset(r'C:\Users\Куликов\yolov8_acne04\data\Annotations\patches.csv', r'C:\Users\Куликов\yolov8_acne04\data\Patches', transform=True)
print(data[1][1])

tens = torch.zeros((1, 3, 1280, 1280))
model = YoloModel(mod = 's')
output = model(tens)
print(output[1][0].shape)