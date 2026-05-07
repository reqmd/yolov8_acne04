import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

from src.utils.transforms import return_transforms

class AcneDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.df = pd.read_csv(csv_path)
        self.images_path = 'data/Patches'
        self.train = train
        self.image_list = self.df['filename'].unique().tolist()
        train_transform, valid_transform = return_transforms()
        self.transform = train_transform if train else valid_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
        img_arr = np.array(image)

        frame = self.df[self.df['filename'] == image_name]
        boxes, labels = [], []
        if len(frame) > 0:
            for _, row in frame.iterrows():
                boxes.append([row['x_center'], row['y_center'], row['width'], row['height']])
                labels.append(row['class_id'])

        transformed = self.transform(image=img_arr, bboxes=boxes, labels=labels)
        image  = transformed['image'].float() / 255.0   # уже тензор после ToTensorV2
        boxes  = transformed['bboxes']
        labels = transformed['labels']

        if len(boxes) > 0:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1] = labels
            targets[:, 2:] = boxes
        else:
            targets = torch.zeros((0, 6))

        return image, targets


def collate_fn(batch):
    images, targets = zip(*batch)
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            t[:, 0] = i
    return torch.stack(images, dim=0), torch.cat(targets, dim=0)