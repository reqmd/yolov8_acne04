import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

from src.utils.transforms import return_transforms

class AcneDataset(Dataset):
    def __init__(self, csv_path, images_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.transform = transform
        self.image_list = os.listdir(self.images_path)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
        img_arr = np.array(image)
        
        # Загрузка ограничивающих рамок и меток класса
        frame = self.df[self.df['filename'] == image_name]
        boxes = []
        labels = []
        if len(frame) > 0:
            for _, row in frame.iterrows():
                x_center, y_center, width, height = row['x_center'], row['y_center'], row['width'], row['height']
                boxes.append([x_center, y_center, width, height])
                labels.append(row['class_id'])

        if self.transform != None:
            self.transform = return_transforms()
            transformed = self.transform(image = img_arr, bboxes = boxes, labels = labels)
        image, boxes, labels = transformed['image'] / 255, transformed['bboxes'], transformed['labels']
        #print(image)
        #print(boxes)
        #print(labels)
        if len(boxes) > 0 and len(labels) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32) 
            labels = torch.tensor(labels, dtype=torch.int8) 
        else:
            boxes = torch.zeros([0, 4], dtype=torch.float32) 
            labels = torch.tensor([], dtype=torch.int8)
        targets = {
                    'boxes': boxes,      # Формат: (N, 4) [x_center, y_center, width, height] в [0, 1]
                    'labels': labels     # Формат: (N,) [class_id1, class_id2, ...]
                }
        return image, targets