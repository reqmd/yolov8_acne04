import os
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
    print(type(y_file))

num = 0

img = Image.open(os.path.join(IMAGES_PATH, y_file['images'][num]['file_name'])).convert('RGB')
img_arr = np.array(img)
orig_img = np.array(img)
prep_img = np.array(img)

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

print(images_list[num])

for item in annot_dict:
    for obj in annot_dict[item]:
        print((annot_dict[item][obj][0], annot_dict[item][obj][3]), (annot_dict[item][obj][2], annot_dict[item][obj][1]))
        cv.rectangle(img_arr, (int(annot_dict[item][obj][0]), int(annot_dict[item][obj][3])), (int(annot_dict[item][obj][2]), int(annot_dict[item][obj][1])), (255, 0, 255), -1)
        cv.rectangle(orig_img, (int(annot_dict[item][obj][0]), int(annot_dict[item][obj][3])), (int(annot_dict[item][obj][2]), int(annot_dict[item][obj][1])), (255, 0, 0), 2)
        cv.rectangle(prep_img, (int(annot_dict[item][obj][0]), int(annot_dict[item][obj][3])), (int(annot_dict[item][obj][2]), int(annot_dict[item][obj][1])), (255, 0, 0), 2)

for annot in y_file['annotations'][:1000]:
    if annot['image_id'] == num:
        xmin = int(annot['coordinates'][0] - annot['radius'])
        xmax = int(annot['coordinates'][0] + annot['radius'])
        ymin = int(annot['coordinates'][1] - annot['radius'])
        ymax = int(annot['coordinates'][1] + annot['radius'])
        roi = img_arr[ymin:ymax, xmin:xmax, :]
        total_size = (ymax - ymin) * (xmax - xmin)
        orig_size = 0
        H, W, _ = roi.shape
        for y in range(H):
            for x in range(W):
                if (roi[y][x][0] == 255) and (roi[y][x][2] == 255) and (roi[y][x][1] == 0):
                    orig_size+=1
        print(f'Площадь перекрытия объекта равна: {orig_size / total_size:.4f}')
        if orig_size / total_size < 0.4:
            cv.rectangle(prep_img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
        cv.rectangle(orig_img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
    elif annot['image_id'] < num:
        continue
    else:
        break
plt.figure(figsize=(12, 12))
plt.axis('off')
plt.imshow(orig_img)

plt.figure(figsize=(12, 12))
plt.axis('off')
plt.imshow(prep_img)

plt.show()