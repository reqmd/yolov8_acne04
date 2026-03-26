import os
import numpy as np
import shutil
import yaml

from PIL import Image
import xml.etree.ElementTree as ET

DATA_PATH = 'data'
IMAGES_PATH = os.path.join(DATA_PATH, 'Classification', 'JPEGImages')
ORIGINAL_ANNOT_PATH = os.path.join(DATA_PATH, 'Detection', 'VOC2007', 'Annotations')

images_list = os.listdir(IMAGES_PATH)
window_size = 640
target_size = 2560

