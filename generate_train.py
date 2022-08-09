# %% Imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
from utils import postprocess_and_save, postprocessmask_and_save

# %% Paramaters
COLS_CSV = ['Loc', 'Images', 'Image_crop', 'Mask_crop', 'Class', 'Aug']
RESIZE_ENABLED = True
RESIZE = (256, 256)
IMAGE_SETS = ['Badalona_SantAdria', 'Castellbisbal', 'Cubelles', 'Gava_Viladecans',
              'Ginestar', 'Hostalric', 'La_Verneda', 'Montornes_del_Valles']
N_IMAGES_SETS = {
    'Badalona_SantAdria': 4,
    'Castellbisbal': 3,
    'Cubelles': 5,
    'Gava_Viladecans': 5,
    'Ginestar': 2,
    'Hostalric': 2,
    'La_Verneda': 2,
    'Montornes_del_Valles': 5,
    'Zona_Franca': 5
}
INPUT_IMAGES_FOLDER = "images"
OUTPUT_FOLDER = "dav_dataset"
OTHERS_CLASS = 5
TH_SIZE = 1000
ASBESTOS = 1
BUILDS = 2

# %% Create folders and CSV and generate dataset
os.makedirs(f"{OUTPUT_FOLDER}/Images", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/Masks", exist_ok=True)

data_csv = pd.DataFrame(columns=COLS_CSV)

for loc in IMAGE_SETS:
    print(f"Processing {loc}, n_images = {N_IMAGES_SETS[loc]}")

    for imgId in range(1, N_IMAGES_SETS[loc] + 1):
        print(f"    Processing Images {imgId} of {N_IMAGES_SETS[loc]}\r", end="")
        img = Image.open(f"{INPUT_IMAGES_FOLDER}/{loc}/Images/image_{imgId}.tif")
        mask = cv2.imread(f"{INPUT_IMAGES_FOLDER}/{loc}/Masks/image_{imgId}_mask.tif", -1).astype(np.uint8)
        img = np.array(img)

        for classId in [ASBESTOS, BUILDS]:
            conn = cv2.connectedComponentsWithStats((mask == classId).astype(np.uint8))
            for i in range(1, conn[0]):
                x, y, w, h, size = conn[2][i]

                if size > TH_SIZE:
                    crop_img = img[y:y + h, x:x + w]
                    crop_mask = mask[y:y + h, x:x + w]
                    #crop_mask = ((conn[1] == i)[y:y + h, x:x + w] * classId).astype(np.uint8)
                    postprocess_and_save(crop_img, f"{OUTPUT_FOLDER}/Images/{loc}_{imgId}_{i}_{classId}.png")
                    postprocessmask_and_save(crop_mask, f"{OUTPUT_FOLDER}/Masks/{loc}_{imgId}_{i}_{classId}.png")

                    data_csv.loc[len(data_csv.index)] = {
                        'Loc': loc,
                        'Images': f'{INPUT_IMAGES_FOLDER}/{loc}/Image_{imgId}.tif',
                        'Image_crop': f"{OUTPUT_FOLDER}/Images/{loc}_{imgId}_{i}_{classId}.png",
                        'Mask_crop': f"{OUTPUT_FOLDER}/Masks/{loc}_{imgId}_{i}_{classId}.png",
                        'Class': classId,
                        'Aug': 0
                    }

data_csv.to_csv(f'{OUTPUT_FOLDER}/dataset.csv')
