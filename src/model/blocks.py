import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel = 1, padding = 0, groups = 1, dilation = 1, need_act = True):
        super().__init__()
        self.need_act = need_act
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              stride=stride, 
                              kernel_size=kernel, 
                              padding=padding,
                              groups = groups,
                              dilation=dilation,
                              bias=False),
        self.bn = nn.BatchNorm2d(num_features=out_channels),
        self.act = nn.SiLU()
    def forward(self, X):
        return self.act(self.bn(self.conv(X))) if self.need_act == True else self.bn(self.conv(X))
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = False):
        super().__init__()
        self.hidden_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels, kernel=3, padding=1, stride=1)
        self.conv2 = ConvBlock(in_channels=self.hidden_channels, out_channels=out_channels, kernel=3, padding=1, stride=1)
        self.shortcut = shortcut

    def forward(self, X):
        residual = X
        out = self.conv2(self.conv1(X))
        return residual + out if self.shortcut == True else out
        
class SPPFBlock():
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super().__init__()
        self.temp_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=self.temp_channels, kernel=(1, 1), stride = 1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding= kernel_size // 2)
        self.conv2 = ConvBlock(in_channels= self.temp_channels * 4, out_channels=out_channels, stride = 1, padding = 1)

    def forward(self, X):
        out = self.conv1(X)
        stage1 = out
        out = self.maxpool(out)
        stage2 = out
        out = self.maxpool(out)
        stage3 = out
        out = self.maxpool(out)
        concat = torch.cat([stage1, stage2, stage3, out], dim = 0)
        return self.conv2(concat)
    
class C2fBlock():
    def __init__():
        return 0
    
class DeteckBlock():
    def __init__():
        return 0

