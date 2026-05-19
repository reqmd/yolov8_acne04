from src.utils.test_utils import load_model, split_into_patches, predict_patch, preprocess_patch
from src.utils.data_utils import make_anchors

import os
from PIL import Image
import torch
from torchvision.ops import nms

TEST_PATH = 'data/Test'
TEST_PATCHES_PATH = 'data/Test/Patches'
device = 'cuda'

model, _ = load_model()

if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

if os.listdir(TEST_PATH) == []:
    print('ВНИМАНИЕ! ПАПКА С ТЕСТОВЫМИ ФОТО ПУСТА')

target_size = 1280
step = 640
for entry in os.listdir(TEST_PATH):
    path = os.path.join(TEST_PATH, entry)
    if os.path.isfile(path):
        image = entry
        break

if not os.path.exists(TEST_PATCHES_PATH):
    os.mkdir(TEST_PATCHES_PATH)

img = Image.open(os.path.join(TEST_PATH, image))
patches = split_into_patches(img = img, image_name=image, target_size=target_size, step=step)

patches_list = os.listdir(TEST_PATCHES_PATH)
anchor_points, stride_tensor = make_anchors(img_size=target_size)
anchor_points = anchor_points.to(device)
stride_tensor = stride_tensor.to(device)

all_boxes = []
all_scores = []

for patch, x_offset, y_offset in patches:
    # Масштаб между патчем и его resize до patch_size
    pw, ph = patch.size
    scale_x = pw / target_size
    scale_y = ph / target_size

    patch_tensor = preprocess_patch(patch, target_size)
    result = predict_patch(model, patch_tensor, anchor_points, stride_tensor, device)

    if len(result['scores']) == 0:
        continue

    # Переводим координаты из патча в координаты полного изображения
    boxes = result['boxes'].cpu()
    boxes[:, 0] = boxes[:, 0] * scale_x + x_offset  # x1
    boxes[:, 1] = boxes[:, 1] * scale_y + y_offset  # y1
    boxes[:, 2] = boxes[:, 2] * scale_x + x_offset  # x2
    boxes[:, 3] = boxes[:, 3] * scale_y + y_offset  # y2

    all_boxes.append(boxes)
    all_scores.append(result['scores'].cpu())

# Объединяем все детекции
all_boxes = torch.cat(all_boxes, dim=0)
all_scores = torch.cat(all_scores, dim=0)

# Финальный NMS по всему изображению — убираем дубли с перекрытий
iou_threshold = 0.45
keep = nms(all_boxes, all_scores, iou_threshold)
final_boxes = all_boxes[keep]
final_scores = all_scores[keep]

print(f'Найдено объектов: {len(final_boxes)}')
    

