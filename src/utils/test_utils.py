from src.model.model import YoloModel
from src.utils.train_utils import post_process
from src.utils.data_utils import decode_predictions

import torch
import numpy as np
from PIL import Image, ImageDraw
from decimal import Decimal
import os
import cv2

def round_custom_decimal(x, threshold=0.5):
    x_dec = Decimal(str(x))
    integer_part = int(x_dec)
    fractional_part = x_dec - integer_part
    if fractional_part >= Decimal(str(threshold)):
        return integer_part + 1
    else:
        return integer_part
    
def load_model(mod='s', device='cuda'):
    model = YoloModel(mod=mod).to(device)
    models_list = os.listdir('models')
    models_with_mod = []
    MODEL_PATH = 'models'
    model_name_end = ''
    for model_name in models_list:
        if model_name.split('8')[1].split('_')[0] == mod:
            models_with_mod.append(model_name)
    ep = 0
    for model_name in models_with_mod:
        epoch = int(model_name.split('_')[3].split('.')[0])
        if epoch > ep:
            model_name_end = model_name
    print(f'Selected model: {model_name_end}')
    model_path = os.path.join(MODEL_PATH, model_name_end)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, ep

def split_into_patches(img, image_name, target_size=1280, step=640):
    img_arr = np.array(img)
    H, W, C = img_arr.shape
    if H >= 640 and H <= 1280 and W >= 640 and W <= 1280:
        H_t, W_t = 1280, 1280
    elif H <= 640 and W <= 640:
        print('Изображения резрешения меньше 640 на 640 не могут быть обработаны')
    else:
        H_t, W_t = round_custom_decimal(H / step) * step, round_custom_decimal(W / step) * step
    img_rsz = cv2.resize(img_arr, (W_t, H_t), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(img_rsz).save(os.path.join('data/Test', f'resized_{image_name}'))
    c = 0
    patches = []
    for h in range(0, H_t - step, step):
        for w in range(0, W_t - step, step):
                x1 = min(w, max(0, W_t - target_size))
                y1 = min(h, max(0, H_t - target_size))
                x2 = x1 + target_size
                y2 = y1 + target_size
                Image.fromarray(img_rsz[y1:y2, x1:x2]).save(os.path.join('data/Test/Patches', f'patch-{c}-{image_name}'))
                c+=1
                patches.append((c, x1, y1)) #patch_idx, xmin, ymin

    seen = set()
    unique_patches = []
    for idx, x1, y1 in patches:
        key = (x1, y1)
        if key not in seen:
            seen.add(key)
            unique_patches.append((idx, x1, y1))
    num_patches = len(unique_patches)
    print(f'Before processing: image {image_name} with resolution {H}, {W}')
    print(f'After processing: image resized_{image_name} with resolution {H_t}, {W_t} divided into {num_patches} patches')
    return unique_patches

def preprocess_patch(patch, patch_size=1280):
    img = patch.resize((patch_size, patch_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    return img.unsqueeze(0) # (1, 3, H, W)

def predict_patch(model, patch_tensor, anchor_points, stride_tensor,
                  conf_threshold=0.25, iou_threshold=0.45, device='cuda'):
    patch_tensor = patch_tensor.to(device)
    with torch.no_grad():
        outputs = model(patch_tensor)
        pred_boxes, pred_cls, _ = decode_predictions(
            outputs, anchor_points, stride_tensor)

    results = post_process(pred_boxes, pred_cls, conf_threshold, iou_threshold)
    return results[0]

def draw_results(image, boxes, scores, save_path='result.jpg'):
    """
    Отрисовывает боксы на изображении.
    """
    draw = ImageDraw.Draw(image)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()

        # Цвет зависит от уверенности
        if score > 0.7:
            color = 'red'
        elif score > 0.5:
            color = 'orange'
        else:
            color = 'yellow'

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), f'{score:.2f}', fill=color)

    image.save(save_path)
    image.show()
    print(f'Результат сохранён: {save_path}')
    return image