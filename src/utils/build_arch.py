from src.model.blocks import ConvBlock, C2fBlock, SPPFBlock, DeteckBlock
import torch.nn as nn
import yaml

BUILD_DICT = {
    'Conv':ConvBlock,
    'C2f':C2fBlock,
    'SPPF':SPPFBlock,
    'Upsample':nn.Upsample,
    'DetectHead':DeteckBlock
}

def build_backbone(path):
    layers = nn.ModuleList()
    with open (path) as file:
        model_config = yaml.safe_load(file)
    for bb_part in model_config['backbone']:
        block_name, params = bb_part[1], bb_part[2]
        block = BUILD_DICT[block_name]
        layers.append(block(*params))
    return layers

class Backbone(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x