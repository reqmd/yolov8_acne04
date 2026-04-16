from src.data.dataset import AcneDataset

dataset = AcneDataset('data/Annotations/patches.csv', r'C:\Users\Куликов\yolov8_acne04\data\Patches', transform=True)
print(dataset[0][1])