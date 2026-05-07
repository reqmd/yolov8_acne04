import albumentations as A
from albumentations.pytorch import ToTensorV2
def return_transforms():
    train_transfrom =  A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.CLAHE(clip_limit=2, tile_grid_size=(6, 6), p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1),
        A.GaussNoise(std_range=(0.01, 0.15), p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Resize(height=1280, width=1280),
        A.Sharpen(alpha = (0.2, 0.4), p = 0.1),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(coord_format='yolo', label_fields=['labels']))

    valid_transform = A.Compose([
        A.Resize(height=1280, width=1280),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(coord_format='yolo', label_fields=['labels']))

    return train_transfrom, valid_transform
