import torch.nn as nn
from src.model.blocks import ConvBlock, SPPFBlock, C2fBlock, DeteckBlock
import yaml 
import math

BUILD_DICT = {
    'Conv':ConvBlock,
    'C2f':C2fBlock,
    'SPPF':SPPFBlock,
    'Upsample':nn.Upsample,
    'DetectHead':DeteckBlock
}

class YoloModel(nn.Module):
    def __init__(self, nc = 1, path = r'C:\Users\Куликов\yolov8_acne04\config\model.yaml', mod = None):
        super().__init__()
        with open(path) as file:
            self.model_config = yaml.safe_load(file)
        keys = self.model_config['scales'].keys()
        if mod not in keys:
            raise ValueError(f'Неверный режим {mod}, доступные: {list(keys)}')
        self.depth, self.width, self.max_ch = self.model_config['scales'][mod]
        self.backbone = self.__build_backbone__()

    def scale_channels(self, ch):
        return min(math.ceil(ch * self.width / 8) * 8, self.max_ch)

    def scale_depth(self, n):
        return max(round(n * self.depth), 1)

    def __build_backbone__(self):
        layers = nn.ModuleList()
        for bb_part in self.model_config['backbone']:
            block_name, params = bb_part[1], bb_part[2].copy()

            if params[0] != 3:  # не трогаем RGB вход
                params[0] = self.scale_channels(params[0])
            params[1] = self.scale_channels(params[1])

            block = BUILD_DICT[block_name]
            layers.append(block(*params))
        return layers

    def forward(self, X):
        for layer in self.backbone:
            X = layer(X)
        return X
    
    