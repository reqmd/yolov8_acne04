import albumentations as A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os
from decimal import Decimal

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Конвертирует YOLO формат в абсолютные координаты bounding box.
    
    Параметры:
    yolo_coords: tuple или list (x_center, y_center, width_norm, height_norm)
    img_width: ширина изображения в пикселях
    img_height: высота изображения в пикселях
    
    Возвращает:
    dict: {'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int}
    """
    if hasattr(yolo_coords, 'values'):
        x_center, y_center, w_norm, h_norm = yolo_coords.values
    else:
        x_center, y_center, w_norm, h_norm = yolo_coords
    
    # Конвертируем в числа (на всякий случай)
    x_center = float(x_center)
    y_center = float(y_center)
    w_norm = float(w_norm)
    h_norm = float(h_norm)
    img_width = float(img_width)
    img_height = float(img_height)
    
    # Конвертируем нормализованные координаты в абсолютные
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    w_abs = w_norm * img_width
    h_abs = h_norm * img_height
    
    # Вычисляем xmin, ymin, xmax, ymax
    xmin = int(x_center_abs - w_abs / 2)
    ymin = int(y_center_abs - h_abs / 2)
    xmax = int(x_center_abs + w_abs / 2)
    ymax = int(y_center_abs + h_abs / 2)
    
    # Обрезаем до границ изображения
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    return xmin, ymin, xmax, ymax

def show_bboxes(num, img_arr, df_annot):
    frag = df_annot[df_annot['id_images'] == num]     
    H, W, C = img_arr.shape       
    for i in range(len(df_annot[df_annot['id_images'] == num])):
        row = frag.iloc[i]
        x = float(row['x_center'])
        y = float(row['y_center'])
        h = float(row['height_norm'])
        w = float(row['width_norm'])
        xmin, ymin, xmax, ymax = yolo_to_bbox((x, y, w, h), img_height=H, img_width=W)
        cv.rectangle(img_arr, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (0, 0, 255), 2)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img_arr)
    return img_arr