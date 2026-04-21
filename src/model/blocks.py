import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 1, stride = 1, padding = 0, groups = 1, dilation = 1, need_act = True):
        super().__init__()
        self.need_act = need_act
        if stride == 2:
            padding = 1
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              stride=stride, 
                              kernel_size=kernel, 
                              padding=padding,
                              groups = groups,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, X):
        out = self.conv(X)
        out = self.bn(out)
        return self.act(out) if self.need_act else out
    
class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = False, kernel_size = 3, expand_ratio = 0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expand_ratio)
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels, kernel=kernel_size, padding=1, stride=1)
        self.conv2 = ConvBlock(in_channels=self.hidden_channels, out_channels=out_channels, kernel=kernel_size, padding=1, stride=1)
        self.shortcut = shortcut

    def forward(self, X):
        residual = X
        out = self.conv2(self.conv1(X))
        return residual + out if self.shortcut == True else out
        
class SPPFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, expand_ratio = 0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expand_ratio)
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels, kernel=(1, 1), stride = 1, need_act=False)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding= kernel_size // 2)
        self.conv2 = ConvBlock(in_channels= self.hidden_channels * 4, out_channels=out_channels, stride = 1)

    def forward(self, X):
        out = self.conv1(X)
        stage1 = out
        out = self.maxpool(out)
        stage2 = out
        out = self.maxpool(out)
        stage3 = out
        out = self.maxpool(out)
        concat = torch.cat([stage1, stage2, stage3, out], dim = 1)
        return self.conv2(concat)
    
class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_bottlenecks = 1, shortcut = False):
        super().__init__()
        self.hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels * 2, stride=1, kernel=1)
        self.conv2 = ConvBlock(in_channels=self.hidden_channels * (2 + n_bottlenecks), out_channels=out_channels, kernel=1, stride=1)
        self.bottlenecks = nn.ModuleList(BottleNeckBlock(in_channels=self.hidden_channels, 
                                                         out_channels=self.hidden_channels, 
                                                         shortcut=shortcut, expand_ratio=1) for _ in range(n_bottlenecks))
    def forward(self, X):
        out = self.conv1(X).split((self.hidden_channels, self.hidden_channels), dim=1)
        out = [out[0], out[1]]
        out.extend(bottleneck(out[-1]) for bottleneck in self.bottlenecks)
        out = torch.cat(out, 1)
        return self.conv2(out)
    
class BoxBranch(nn.Module):
    def __init__(self, in_channels, reg_max = 16):
        super().__init__()
        self.hidden_channels = max(in_channels, 4 * reg_max)
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels, kernel=3, stride = 1, padding = 1),
            ConvBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels= 4 * reg_max, kernel_size=1)
        )
    def forward(self, X):
        return self.conv(X)
    
class ClsBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.hidden_channels = max(in_channels, 32)
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=self.hidden_channels, kernel=3, stride = 1, padding = 1),
            ConvBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels= 1, kernel_size=1)
        )
    def forward(self, X):
        return self.conv(X)
    
class DFLConvert(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.weight.data = torch.arange(
            reg_max, dtype=torch.float
        ).reshape(1, reg_max, 1, 1)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        B, _, A = x.shape
        x = x.view(B, 4, self.reg_max, A)
        x = x.softmax(2)           
        x = self.conv(x.permute(0, 1, 3, 2).reshape(B * 4, self.reg_max, A, 1))
        return x.reshape(B, 4, A)  
    
class DeteckBlock(nn.Module):
    def __init__(self, in_channels, reg_max = 16):
        super().__init__()
        self.reg_max = reg_max
        self.box_branch = BoxBranch(in_channels=in_channels, reg_max=reg_max)
        self.cls_branch = ClsBranch(in_channels=in_channels)
        self.dfl = DFLConvert(reg_max)

    def forward(self, X, inference = False):
        B = X.shape[0]
        box = self.box_branch(X)          
        cls = self.cls_branch(X)          
        box = box.view(B, 4 * self.reg_max, -1)  
        cls = cls.view(B, 1, -1)                  
        if inference:
            box = self.dfl(box)
            cls = cls.sigmoid()
        return box, cls

