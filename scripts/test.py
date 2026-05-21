from src.utils.test_utils import load_model, split_into_patches, predict_patch, preprocess_patch, draw_results
from src.utils.data_utils import make_anchors
from src.utils.train_utils import load_params, evaluate
from src.data.dataset import AcneDataset, collate_fn

import os
from PIL import Image
import torch
from torchvision.ops import nms
from torch.utils.data import DataLoader

TEST_PATH = 'data/Test'
TEST_PATCHES_PATH = 'data/Test/Patches'
YAML_ROOT = 'config/preprocessing.yaml'

device = 'cuda'
model, _ = load_model()

epochs, batch_size, lr, weight_decay, num_workers, pin_memory, persistent_workers, mod, device = load_params(YAML_ROOT)
val_data = AcneDataset('data/Annotations/val.csv', train=False)
val_loader = DataLoader(dataset=val_data, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=collate_fn, 
                        num_workers=num_workers, 
                        pin_memory=pin_memory, 
                        persistent_workers=persistent_workers)


if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

if os.listdir(TEST_PATH) == []:
    raise FileNotFoundError('ВНИМАНИЕ! ПАПКА С ТЕСТОВЫМИ ФОТО ПУСТА')

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
NEED_TO_OUT_MAP = False
if NEED_TO_OUT_MAP:
    map_result = evaluate(model, val_loader, anchor_points, stride_tensor)
    print(f'mAP@50:    {map_result["map_50"]:.4f}')
    print(f'mAP@50-95: {map_result["map"]:.4f}')

for idx, x_offset, y_offset in patches:
    patch = Image.open(os.path.join(TEST_PATCHES_PATH, f'patch-{idx-1}-{image}'))
    patch_tensor = preprocess_patch(patch, target_size)
    result = predict_patch(model, patch_tensor, anchor_points, stride_tensor)

    if len(result['scores']) == 0:
        continue
    # Переводим координаты из патча в координаты полного изображения
    boxes = result['boxes'].cpu()
    boxes[:, 0] = boxes[:, 0] + x_offset  # x1
    boxes[:, 1] = boxes[:, 1] + y_offset  # y1
    boxes[:, 2] = boxes[:, 2] + x_offset  # x2
    boxes[:, 3] = boxes[:, 3] + y_offset  # y2

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
resized_image = Image.open(os.path.join(TEST_PATH, f'resized_{image}'))
draw_results(resized_image, final_boxes, final_scores, save_path='result.jpg')
    

