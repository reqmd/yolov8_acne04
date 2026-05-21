import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import cv2 as cv
import yaml
import xml.etree.ElementTree as ET

#Пути данных
DATA_PATH = 'data'
CLASSIFICATION_PATH = os.path.join(DATA_PATH, 'Classification')
DETECTION_PATH = os.path.join(DATA_PATH, 'Detection')

#Список изображений
images_list = os.listdir(os.path.join(CLASSIFICATION_PATH, 'JPEGImages'))

#Файл дополнительной разметки в формате словаря
with open(os.path.join('data', 'acne04v2-main', 'Acne04-v2_annotations.json'), 'r') as file:
    y_file = yaml.safe_load(file)
    print(f'The additional markup file has been uploaded successfully')

#Датафрейм с дополнительной разметкой 
df = pd.DataFrame(y_file['annotations'])
df_images = pd.DataFrame(y_file['images'])
# Фильтрация от ненужных меток 
idx = 0
num_add = 0
threshhold = 0.5

#Создание набора данных после фильтрации
df_end = pd.DataFrame(columns=df.columns)
for num in range(len(pd.unique(df['image_id']))):
    if num % 250 == 0:
        print(f'Working with an image {num}')
    img = Image.open(os.path.join(CLASSIFICATION_PATH, 'JPEGImages', df_images[df_images['id'] == num]['file_name'].values[0])).convert('RGB')
    img_arr = np.array(img)
    H, W, C = img_arr.shape
    mask_arr = np.zeros((H, W, C))

    annot_dict = {}
    image = y_file['images'][num]['file_name']
    name, _ = image.split('.')
    tree = ET.parse(os.path.join(DETECTION_PATH, 'VOC2007', 'Annotations', f'{name}.xml'))
    root = tree.getroot()
    objects = {}
    for i, item in enumerate(root):
        if item.find('bndbox') != None:
            xmin = item.find('bndbox/xmin').text
            ymin = item.find('bndbox/ymin').text
            xmax = item.find('bndbox/xmax').text
            ymax = item.find('bndbox/ymax').text
            objects[i] = [xmin, ymin, xmax, ymax]

    annot_dict[name] = objects
    
    for item in annot_dict:
        for obj in annot_dict[item]:
            cv.rectangle(mask_arr, (int(annot_dict[item][obj][0]), int(annot_dict[item][obj][3])), (int(annot_dict[item][obj][2]), int(annot_dict[item][obj][1])), (255, 255, 255), -1)
    L = len(df[df['image_id'] == num]) + idx
    for n in range(idx, L):
        #print(n, num, L)
        radius = df['radius'][n]
        coords = df['coordinates'][n]
        xmin = int(coords[0] - radius)
        xmax = int(coords[0] + radius)
        ymin = int(coords[1] - radius)
        ymax = int(coords[1] + radius)
        roi = mask_arr[ymin:ymax, xmin:xmax, :]
        total_size = (ymax - ymin) * (xmax - xmin)
        orig_size = 0
        H, W, _ = roi.shape
        for y in range(H):
            for x in range(W):
                if (roi[y][x][0] == 255) and (roi[y][x][2] == 255) and (roi[y][x][1] == 255):
                    orig_size+=1
        #print(f'Площадь перекрытия объекта равна: {orig_size / total_size:.4f}')
        if orig_size / total_size <= threshhold:
            df_end.loc[num_add] = df.loc[n].copy()
            num_add+=1
    #print(num_add, idx, n)
    idx+=len(df[df['image_id'] == num])
    if num % 250 == 0:
        print(f'Were added:{num_add} out of {L}')

print(f'The total number of objects after filtering:{num_add} out of {L}')

#Создание первого csv файла с данными о изображениях
col_list = ['id_images', 'filename', 'height', 'width']
df_images = pd.DataFrame(columns=col_list)
if not os.path.exists('data/Annotations'):
    os.mkdir('data/Annotations')
for i, image in enumerate(images_list):
    img = Image.open(os.path.join(CLASSIFICATION_PATH, 'JPEGImages', image)).convert('RGB')
    img_arr = np.array(img)
    H, W, _ = img_arr.shape
    df_images.loc[i] = pd.Series([i, image, H, W], index = df_images.columns)
print('The .csv file with image markup was successfully createdн')

#Создание первого датафрейма из оригинальной разметки
annot_columns = ['id_images', 'xmin', 'xmax', 'ymin', 'ymax', 'annot_V']
df_v1 = pd.DataFrame(columns=annot_columns)
counter = 0
for image in images_list:
    num = df_images[df_images['filename'] == image]['id_images']
    H = df_images[df_images['filename'] == image]['height']
    W = df_images[df_images['filename'] == image]['width']
    name, _ = image.split('.')
    tree = ET.parse(os.path.join(DETECTION_PATH, 'VOC2007', 'Annotations', f'{name}.xml'))
    root = tree.getroot()
    columns = ['xmin', 'ymin', 'xmax', 'ymax']
    for item in root:
        if item.find('bndbox') != None:
            xmin = float(item.find('bndbox/xmin').text) 
            ymin = float(item.find('bndbox/ymin').text) 
            xmax = float(item.find('bndbox/xmax').text) 
            ymax = float(item.find('bndbox/ymax').text) 
            df_v1.loc[counter] = pd.Series([num.values[0], xmin, xmax, ymin, ymax, 'V1'], index = df_v1.columns)
            counter+=1
print('The first dataframe for v1 has been created')

#Создание второго датафрейма из дополнительной разметки
df_images_v2 = pd.DataFrame(y_file['images'])
df_end = df_end.drop('class_name', axis = 1)
df_end = df_end.drop('id', axis = 1)
df_v2 = pd.DataFrame(columns=annot_columns)
c = len(df_v1)
for name in df_images['filename']:
    id = df_images_v2[df_images_v2['file_name'] == name]['id']
    id_v1 = df_images[df_images['filename'] == name]['id_images']

    if id.values.size != 0:
        id = id.values[0]
        id_v1 = id_v1.values[0]
        fragment = df_end[df_end['image_id'] == id]
        H = df_images[df_images['id_images'] == id]['height'].values[0]
        W = df_images[df_images['id_images'] == id]['width'].values[0]
        for n in range(len(fragment)):
            row = fragment.iloc[n]
            radius = row['radius']
            coords = row['coordinates']
            xmin = float(coords[0] - radius) 
            xmax = float(coords[0] + radius) 
            ymin = float(coords[1] - radius) 
            ymax = float(coords[1] + radius) 
            df_v2.loc[c] = pd.Series([id_v1, xmin, xmax, ymin, ymax, 'V2'], index = df_v2.columns)
            c+=1
print('The second dataframe for v2 has been created')

#Слияние двух датафреймов в один и фильтрация от больших и отрицательных значений координат  
df_FINAL = pd.concat((df_v1, df_v2))
df_FINAL = df_FINAL[(df_FINAL['xmax'] < 7000) & (df_FINAL['xmin'] > 0)]
df_sorted = df_FINAL.sort_values(by = 'id_images', )

df_merged = df_sorted.merge(df_images[['id_images', 'height', 'width']], 
                            on='id_images', 
                            how='left')

#Проверяем, что все объекты получили размеры
if df_merged['width'].isna().sum() > 0:
    print(f"Warning: {df_merged['width'].isna().sum()} objects without info about size!")
    df_merged = df_merged.dropna(subset=['width', 'height'])

#Нормализуем координаты (каждая строка использует свои width/height)
df_merged['x_center'] = ((df_merged['xmin'] + df_merged['xmax']) / 2) / df_merged['width']
df_merged['y_center'] = ((df_merged['ymin'] + df_merged['ymax']) / 2) / df_merged['height']
df_merged['width_norm'] = (df_merged['xmax'] - df_merged['xmin']) / df_merged['width']
df_merged['height_norm'] = (df_merged['ymax'] - df_merged['ymin']) / df_merged['height']

#Обрезаем до [0, 1] на случай выхода за границы
df_merged['x_center'] = df_merged['x_center'].clip(0, 1)
df_merged['y_center'] = df_merged['y_center'].clip(0, 1)
df_merged['width_norm'] = df_merged['width_norm'].clip(0, 1)
df_merged['height_norm'] = df_merged['height_norm'].clip(0, 1)

print('The combined markup file was successfully compiled')

#Отображение изображения с разметкой 
num = 1168
frag = df_sorted[df_sorted['id_images'] == num]            
img = Image.open(os.path.join(CLASSIFICATION_PATH, 'JPEGImages', df_images[df_images['id_images'] == num].values[0][1])).convert('RGB')
H = df_images[df_images['id_images'] == num].values[0][2]
W = df_images[df_images['id_images'] == num].values[0][3]
img_arr = np.array(img)
for i in range(len(df_sorted[df_sorted['id_images'] == num])):
    row = frag.iloc[i]
    xmin = int(row['xmin']) #* H
    xmax = int(row['xmax']) #* H
    ymin = int(row['ymin']) #* W
    ymax = int(row['ymax']) #* W
    cv.rectangle(img_arr, (xmin, ymax), (xmax, ymin), (255, 0, 0), 2)
plt.figure(figsize=(12, 12))
plt.axis('off')
plt.imshow(img_arr)
#Image.fromarray(img_arr).save('example.jpg')
images_path = 'data/Annotations/images.csv'
annot_path = 'data/Annotations/annot.csv'
df_images.to_csv(images_path)
df_merged.to_csv(annot_path)
print(f'Files with image and annotation markup were saved in the path {images_path}, {annot_path}')

v1_path = 'data/Detection'
v2_path = 'data/acne04v2-main'
if os.path.exists(v1_path) and os.path.exists(v2_path):
    shutil.rmtree(v1_path)
    shutil.rmtree(v2_path)
    print(f'Folders {v1_path} and {v2_path} has been deleted')