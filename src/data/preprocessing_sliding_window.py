import albumentations as A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os
from decimal import Decimal
from sklearn.model_selection import train_test_split

#from utils.show_bboxes import show_bboxes

TRAIN_RATIO = 0.8
RANDOM_STATE = 101

def round_custom_decimal(x, threshold=0.5):
    x_dec = Decimal(str(x))
    integer_part = int(x_dec)
    fractional_part = x_dec - integer_part
    if fractional_part >= Decimal(str(threshold)):
        return integer_part + 1
    else:
        return integer_part

DATA_PATH = 'data'
CLASSIFICATION_PATH = os.path.join(DATA_PATH, 'Classification', 'JPEGImages')
ANNOT_PATH = os.path.join(DATA_PATH, 'Annotations')

df_images = pd.read_csv(os.path.join(ANNOT_PATH, 'images.csv'))
df_annot = pd.read_csv(os.path.join(ANNOT_PATH, 'annot.csv'))
if not os.path.exists(os.path.join(ANNOT_PATH, 'patches.csv')):
    df_patches = pd.DataFrame(columns=['filename', "images_id", "patch_id", "x_center", "y_center", "width", "height", 'class_id'])

if not os.path.exists('data/Patches'):
    os.mkdir('data/Patches')
    
if not os.path.exists('data/Test'):
    os.mkdir('data/Test')

target_size = 1280
step = 640
images_list = os.listdir(CLASSIFICATION_PATH)


g_c = 0
for num in range(len(images_list)):
    image_name = images_list[num].split('.')[0]
    image = Image.open(os.path.join(CLASSIFICATION_PATH, images_list[num]))
    img_arr = np.array(image)
    
    He, Wi, id = df_images[df_images['filename'] == images_list[num]]['height'], df_images[df_images['filename'] == images_list[num]]['width'], df_images[df_images['filename'] == images_list[num]]['id_images']
    H, W, C = img_arr.shape

    H_t, W_t = round_custom_decimal(H / step) * step, round_custom_decimal(W / step) * step
    if H_t <= 1280 and W_t <= 1280:
        continue

    img_rsz = cv.resize(img_arr, (H_t, W_t))
    H_t, W_t, _ = img_rsz.shape
    #print(img_rsz.dtype)
    print(image_name)
    print(H_t, W_t)
    print('/////////////////////////////////')
    frag = df_annot[df_annot['id_images'] == id.values[0]]
    # Функция для разметки bboxes на изображении
    #img_rsz = show_bboxes(num=num, img_arr=img_rsz, df_annot=df_annot)
    if H_t <= 1280 and W_t <= 1280:
        continue
    else:
        c = 0
        for h in range(0, H_t - step, step):
            for w in range(0, W_t - step, step):
                for idx, row in frag.iterrows():
                    x_center, y_center, width, height = row['x_center'], row['y_center'], row['width_norm'], row['height_norm']
                    x_min = (x_center - width / 2) * W_t
                    x_max = (x_center + width / 2) * W_t
                    y_min = (y_center - height / 2) * H_t
                    y_max = (y_center + height / 2) * H_t
                    if (x_min >= w and x_max <= w + target_size and
                        y_min >= h and y_max <= h + target_size):
                        new_x_abs = x_min - w
                        new_y_abs = y_min - h
                        new_width_abs = x_max - x_min
                        new_height_abs = y_max - y_min

                        # Относительные координаты для патча (YOLO формат)
                        new_x_center = (new_x_abs + new_width_abs / 2) / target_size
                        new_y_center = (new_y_abs + new_height_abs / 2) / target_size
                        new_width = new_width_abs / target_size
                        new_height = new_height_abs / target_size
                        df_patches.loc[g_c] = pd.Series([f'{image_name}-{c}.jpg', id.values[0], c, new_x_center, new_y_center, new_width, new_height, 0], index = df_patches.columns)
                        g_c+=1
                Image.fromarray(img_rsz[h:h+target_size, w:w+target_size]).save(os.path.join('data/Patches', f'{image_name}-{c}.jpg'))
                c+=1
df_patches.to_csv(os.path.join(ANNOT_PATH, 'patches.csv'), index=False)

unique_images = df_patches['filename'].unique()
train_images, val_images = train_test_split(
    unique_images,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_STATE
)
train_df = df_patches[df_patches['filename'].isin(train_images)]
val_df   = df_patches[df_patches['filename'].isin(val_images)]
train_df.to_csv(os.path.join(ANNOT_PATH, 'train.csv'), index=False)
val_df.to_csv(os.path.join(ANNOT_PATH, 'val.csv'),   index=False)
print(f'Train: {len(train_images)} изображений')
print(f'Val:   {len(val_images)} изображений')