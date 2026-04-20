from src.data.dataset import AcneDataset
from src.model.blocks import ConvBlock, BottleNeckBlock, SPPFBlock, C2fBlock, DeteckBlock

import torch 
from torchsummary import summary
conv = DeteckBlock(in_channels=256).to('cuda')

print(summary(conv, (256, 80, 80)))