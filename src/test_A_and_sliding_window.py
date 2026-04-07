import albumentations as A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os
from decimal import Decimal

from utils.show_bboxes import show_bboxes

def round_custom_decimal(x, threshold=0.6):
    x_dec = Decimal(str(x))
    integer_part = int(x_dec)
    fractional_part = x_dec - integer_part
    if fractional_part >= Decimal(str(threshold)):
        return integer_part + 1
    else:
        return integer_part

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # Рисуем вертикальные линии
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # Рисуем горизонтальные линии
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv.line(img, (0, y), (w, y), color=color, thickness=thickness)

DATA_PATH = 'data'
CLASSIFICATION_PATH = os.path.join(DATA_PATH, 'Classification', 'JPEGImages')
ANNOT_PATH = os.path.join(DATA_PATH, 'Annotations')

target_size = 640

num = 1
images_list = os.listdir(CLASSIFICATION_PATH)
image = Image.open(os.path.join(CLASSIFICATION_PATH, images_list[num]))
img_arr = np.array(image)

df_images = pd.read_csv(os.path.join(ANNOT_PATH, 'images.csv'))
df_annot = pd.read_csv(os.path.join(ANNOT_PATH, 'annot.csv'))

He, Wi, id = df_images[df_images['filename'] == images_list[num]]['height'], df_images[df_images['filename'] == images_list[num]]['width'], df_images[df_images['filename'] == images_list[num]]['id_images']
H, W, C = img_arr.shape
H_t, W_t = round_custom_decimal(H / target_size) * target_size, round_custom_decimal(W / target_size) * target_size
img_rsz = cv.resize(img_arr, (H_t, W_t))
frag = df_annot[df_annot['id_images'] == id.values[0]]
print(int(He.values[0]), int(Wi.values[0]))
print(H, W)
print(H / target_size , W / target_size)
print(H_t, W_t)

show_bboxes(num=num, img_arr=img_arr, df_annot=df_annot)
show_bboxes(num=num, img_arr=img_rsz, df_annot=df_annot)
draw_grid(img=img_rsz, grid_shape=(round_custom_decimal(H / target_size), round_custom_decimal(W / target_size)), thickness=2)
plt.show()
# cv.namedWindow('HUITA', cv.WINDOW_NORMAL)
# cv.imshow('HUITA', img_rsz)
# cv.waitKey(0)  # Ожидание нажатия любой клавиши
# cv.destroyAllWindows()
Image.fromarray(img_rsz).save('image.jpg')