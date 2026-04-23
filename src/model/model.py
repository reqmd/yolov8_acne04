import torch.nn as nn
from src.model.blocks import ConvBlock, SPPFBlock, C2fBlock, DetectHead
import yaml 
import math
import torch

BUILD_DICT = {
    'Conv':ConvBlock,
    'C2f':C2fBlock,
    'SPPF':SPPFBlock,
    'nn.Upsample':nn.Upsample,
    'DetectHead':DetectHead
}
NO_SCALE_BLOCKS = {'nn.Upsample', 'DetectHead', 'Concat'}

class YoloModel(nn.Module):
    def __init__(self, mod = None, nc = 1, path = r'C:\Users\Куликов\yolov8_acne04\config\model.yaml'):
        super().__init__()
        with open(path) as file:
            self.model_config = yaml.safe_load(file)
        keys = self.model_config['scales'].keys()
        if mod not in keys:
            raise ValueError(f'Неверный режим {mod}, доступные: {list(keys)}')
        self.depth, self.width, self.max_ch = self.model_config['scales'][mod]
        self.backbone = self.__build_backbone__()
        self.head = self.__build_head__()

    def scale_channels(self, ch):
        return min(math.ceil(ch * self.width / 8) * 8, self.max_ch)

    def scale_depth(self, n):
        return max(round(n * self.depth), 1)

    def __build_backbone__(self):
        layers = nn.ModuleList()
        for bb_part in self.model_config['backbone']:
            block_name, params = bb_part[1], bb_part[2].copy()
            if params[0] != 3:
                params[0] = self.scale_channels(params[0])
            params[1] = self.scale_channels(params[1])
            if block_name == 'C2f':
                params[2] = self.scale_depth(params[2])
            block = BUILD_DICT[block_name]
            layers.append(block(*params))
        return layers
    
    def __build_head__(self):
        layers = nn.ModuleList()
        self.concat_sources = {}
        self.detect_sources = None

        self.layer_channels = {}
        for i, bb_part in enumerate(self.model_config['backbone']):
            block_name, params = bb_part[1], bb_part[2].copy()
            out_ch = self.scale_channels(params[1]) if params[1] != 3 else params[1]
            self.layer_channels[i] = out_ch

        offset = len(self.model_config['backbone'])
        prev_ch = self.layer_channels[offset - 1]

        for i, bb_part in enumerate(self.model_config['head']):
            output_from, block_name, params = bb_part[0], bb_part[1], bb_part[2].copy()
            global_i = offset + i

            if block_name == 'Concat':
                sources = [global_i - 1 if idx == -1 else idx for idx in output_from]
                self.concat_sources[global_i] = sources
                prev_ch = sum(self.layer_channels[idx] for idx in sources)
                self.layer_channels[global_i] = prev_ch
                layers.append(nn.Identity())
                continue

            if block_name in ('nn.Upsample', 'Upsample'):
                _, scale_factor, mode = params
                self.layer_channels[global_i] = prev_ch
                layers.append(nn.Upsample(scale_factor=scale_factor, mode=mode))
                continue

            if block_name == 'DetectHead':
                self.detect_sources = output_from  # [15, 18, 21] 
                in_channels_list = [self.layer_channels[idx] for idx in output_from]
                layers.append(DetectHead(in_channels_list))
                continue
            out_ch = self.scale_channels(params[1])

            if block_name == 'C2f':
                n = self.scale_depth(params[2])
                shortcut = params[3] if len(params) > 3 else False
                layers.append(C2fBlock(prev_ch, out_ch, n, shortcut))

            elif block_name == 'Conv':
                layers.append(ConvBlock(prev_ch, out_ch, *params[2:]))

            self.layer_channels[global_i] = out_ch
            prev_ch = out_ch
        return layers

    def forward(self, X):
        outputs = {}
        # Backbone part
        for i, layer in enumerate(self.backbone):
            X = layer(X)
            outputs[i] = X
            #print(f"backbone[{i}] {type(layer).__name__}: {X.shape}")
        
        # Head part
        offset = len(self.backbone)
        for i, layer in enumerate(self.head):
            global_i = offset + i
            if isinstance(layer, nn.Identity):  # Concat
                src_indices = self.concat_sources[global_i]
                tensors = [outputs[idx] for idx in src_indices]
                X = torch.cat(tensors, dim=1)
                #print(f"head[{i}] Concat(sources={src_indices}): {X.shape}")
            elif isinstance(layer, DetectHead):
                src_indices = self.detect_sources  # [15, 18, 21]
                p3 = outputs[src_indices[0]]
                p4 = outputs[src_indices[1]]
                p5 = outputs[src_indices[2]]
                #print(f"head[{i}] DetectHead: p3={p3.shape} p4={p4.shape} p5={p5.shape}")
                X = layer([p3, p4, p5])
            else:
                X = layer(X)
                #print(f"head[{i}] {type(layer).__name__}: {X.shape}")
            outputs[global_i] = X
        return X