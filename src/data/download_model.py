import gdown
import os
def download_model(model_name = 'Yolov8s200.pth'):
    if not os.path.exists(model_name) and not os.path.exists('models'):
        url = 'https://drive.google.com/uc?id=1SLQLuxbJ4_irCUFW8C0N0rXPQGbjZjoC'
        output_path = model_name
        gdown.download(url, output_path, quiet=False)
        print(f"The file has been downloaded: {output_path}")
        os.mkdir('models')
        os.rename(model_name, f'models/{model_name}')