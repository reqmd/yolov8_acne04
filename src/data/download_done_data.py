import gdown
import os
import zipfile
if not os.path.exists('archive.zip') and not os.path.exists('data'):
    url = 'https://drive.google.com/uc?id=1szUaUjghGUbWRfDOgkelSnBmWdjaG6NL'
    output_path = 'archive.zip'
    gdown.download(url, output_path, quiet=False)
    print(f"Файл скачан: {output_path}")

    with zipfile.ZipFile('archive.zip', 'r') as z_file:
        z_file.extractall('data')
        print('Файл успешно разархивирован')
    os.remove('archive.zip')