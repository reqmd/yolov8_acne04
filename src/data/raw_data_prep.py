import os
import pandas as pd
import numpy as np
import yaml
import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image
import xml.etree.ElementTree as ET


DATA_PATH = 'data'
IMAGES_PATH = os.path.join(DATA_PATH, 'Classification', 'JPEGImages')
ORIGINAL_ANNOT_PATH = os.path.join(DATA_PATH, 'Detection', 'VOC2007', 'Annotations')

images_list = os.listdir(IMAGES_PATH)
window_size = 640
target_size = 2560

with open(os.path.join(DATA_PATH, 'acne04v2-main', 'Acne04-v2_annotations.json'), 'r') as file:
    y_file = yaml.safe_load(file)
    print(f'Файл дополнительной разметки прочитан, он имеет тип {type(y_file)}')
    df = pd.DataFrame(y_file['annotations'])

df_end = pd.DataFrame(columns=df.columns)

idx = 0
for num in range(len(pd.unique(df['image_id']))):
    img = Image.open(os.path.join(IMAGES_PATH, y_file['images'][num]['file_name'])).convert('RGB')
    img_arr = np.array(img)
    H, W, C = img_arr
    mask_arr = np.zeros((H, W, C))

    annot_dict = {}
    image = y_file['images'][num]['file_name']
    name, _ = image.split('.')
    tree = ET.parse(os.path.join(ORIGINAL_ANNOT_PATH, f'{name}.xml'))
    root = tree.getroot()
    columns = ['xmin', 'ymin', 'xmax', 'ymax']
    c=0
    objects = {}
    for item in root:
        if item.find('bndbox') != None:
            c+=1
            xmin = item.find('bndbox/xmin').text
            ymin = item.find('bndbox/ymin').text
            xmax = item.find('bndbox/xmax').text
            ymax = item.find('bndbox/ymax').text
            objects[c] = [xmin, ymin, xmax, ymax]
    annot_dict[name] = objects
    
    for item in annot_dict:
        for obj in annot_dict[item]:
            print((annot_dict[item][obj][0], annot_dict[item][obj][3]), (annot_dict[item][obj][2], annot_dict[item][obj][1]))
            cv.rectangle(mask_arr, (int(annot_dict[item][obj][0]), int(annot_dict[item][obj][3])), (int(annot_dict[item][obj][2]), int(annot_dict[item][obj][1])), (255, 255, 255), -1)
    num_add = 0
    for n in range(len(df[df['image_id'] == num])):
        radius = df['radius'][n]
        coords = df['coordinates'][n]
        xmin = int(coords[0] - radius)
        xmax = int(coords[0] + radius)
        ymin = int(coords[1] - radius)
        ymax = int(coords[1] + radius)
        roi = img_arr[ymin:ymax, xmin:xmax, :]
        total_size = (ymax - ymin) * (xmax - xmin)
        orig_size = 0
        H, W, _ = roi.shape
        for y in range(H):
            for x in range(W):
                if (roi[y][x][0] == 255) and (roi[y][x][2] == 255) and (roi[y][x][1] == 255):
                    orig_size+=1
        #print(f'Площадь перекрытия объекта равна: {orig_size / total_size:.4f}')
        if orig_size / total_size < 0.4:
            num_add+=1
            df_end = df_end.append(df[idx + n])
    idx+=len(df[df['image_id'] == num])
    print(f'Было удалено:{num_add} из {len(df[df['image_id'] == num])}')