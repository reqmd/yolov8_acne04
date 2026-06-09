import gdown
import os
import zipfile
def download_data():
    if not os.path.exists('archive.zip') and not os.path.exists('data'):
        url = 'https://drive.google.com/uc?id=1di4kqMbT1kNSessd2jqVeHR1Op-_2xqR'
        output_path = 'archive.zip'
        gdown.download(url, output_path, quiet=False)
        print(f"The file has been downloaded: {output_path}")

        with zipfile.ZipFile('archive.zip', 'r') as z_file:
            z_file.extractall()
            print('The file has been successfully unzipped')
        os.remove('archive.zip')