from src.data.dataset import AcneDataset
from src.model.blocks import ConvBlock, BottleNeck, SPPFBlock

import torch 
from torchsummary import summary
conv = SPPFBlock(in_channels=128, out_channels=128).to('cuda')

print(summary(conv, (128, 80, 80)))