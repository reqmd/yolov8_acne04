from src.data.dataset import AcneDataset
from src.model.blocks import ConvBlock, BottleNeckBlock, SPPFBlock, C2fBlock, DeteckBlock
from src.model.test_losses import test_ciou_loss, test_dfl_loss
from src.model.model import YoloModel
from torchsummary import summary

model = YoloModel(mod='x').to('cuda')
summary(model = model, input_size=(3, 1280, 1280))
