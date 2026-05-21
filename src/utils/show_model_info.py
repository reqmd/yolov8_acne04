from src.model.model import YoloModel
from torchinfo import summary

def show_model_info(mod = 'n'):
    """
    Показывает параметры модели с помощью torchinfo.summary

    Параметры:
    mod: режим YOLO
    """
    model = YoloModel(mod=mod).to('cuda')
    summary(model = model, input_size=(1, 3, 1280, 1280))

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params     = sum(p.numel() for p in model.head.parameters())
    print(f"Backbone: {backbone_params:,}")
    print(f"Head:     {head_params:,}")

    for i, block in enumerate(model.head[-1].heads):
        params = sum(p.numel() for p in block.parameters())
        box_params = sum(p.numel() for p in block.box_branch.parameters())
        cls_params = sum(p.numel() for p in block.cls_branch.parameters())
        print(f"DeteckBlock[{i}]: total={params:,}  box={box_params:,}  cls={cls_params:,}")