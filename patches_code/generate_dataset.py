# %% Imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import rasterio
import os

# %% Paramaters
COLS_CSV = ['Loc', 'Images', 'Image_crop', 'Mask_crop', 'Col', 'Row', 'X', 'Y',
            'Asbestos', 'Streets', 'Buildings', 'GreenSpaces', 'Others']
CROP_WIDTH = 10#Last 10, 19
CROP_HEIGHT = 10#Last 10, 19
RESIZE_ENABLED = False
RESIZE = (256, 256)
IMAGE_SETS = ['Badalona_SantAdria', 'Castellbisbal', 'Cubelles', 'Gava_Viladecans',
              'Ginestar', 'Hostalric', 'La_Verneda', 'Montornes_del_Valles', 'Zona_Franca']
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
OUTPUT_FOLDER = "dataset"
OTHERS_CLASS = 5

# %% Create folders and CSV and generate dataset
try:
    os.mkdir(OUTPUT_FOLDER)
    os.mkdir(f"{OUTPUT_FOLDER}/Images")
    os.mkdir(f"{OUTPUT_FOLDER}/Masks")
except:
    pass

data_csv = pd.DataFrame(columns=COLS_CSV)


def postprocessmask_and_save(image, outfile):
    image[image == 127] = OTHERS_CLASS
    image[image == 0] = OTHERS_CLASS
    if RESIZE_ENABLED:
        image = cv2.resize(image, RESIZE, interpolation=cv2.INTER_NEAREST)

    Image.fromarray(image).save(outfile)


def postprocess_and_save(image, outfile):
    if RESIZE_ENABLED:
        image = cv2.resize(image, RESIZE)

    Image.fromarray(image).save(outfile)


for loc in IMAGE_SETS:
    print(f"Processing {loc}, n_images = {N_IMAGES_SETS[loc]}")

    for imgId in range(1, N_IMAGES_SETS[loc] + 1):
        print(f"    Processing Images {imgId} of {N_IMAGES_SETS[loc]}\r", end="")
        img = Image.open(f"{INPUT_IMAGES_FOLDER}/{loc}/Images/image_{imgId}.tif")
        img_tif = rasterio.open(f"{INPUT_IMAGES_FOLDER}/{loc}/Images/image_{imgId}.tif")
        mask = cv2.imread(f"{INPUT_IMAGES_FOLDER}/{loc}/Masks/image_{imgId}_mask.tif", -1).astype(np.uint8)
        img = np.array(img)

        h, w = img.shape[0:2]
        h_crop, w_crop = h // CROP_HEIGHT, w // CROP_WIDTH

        for i in range(CROP_HEIGHT):
            for j in range(CROP_WIDTH):
                start_h, start_w = i * h_crop, j * w_crop
                crop_img = img[start_h: start_h + h_crop, start_w: start_w + w_crop]
                crop_mask = mask[start_h: start_h + h_crop, start_w: start_w + w_crop]
                postprocess_and_save(crop_img, f"{OUTPUT_FOLDER}/Images/{loc}_{imgId}_{i}_{j}.jpg")
                postprocessmask_and_save(crop_mask, f"{OUTPUT_FOLDER}/Masks/{loc}_{imgId}_{i}_{j}.jpg")

                # In this api X is the height and Y the width.
                x, y = img_tif.xy(start_h, start_w, offset='ul')

                data_csv.loc[len(data_csv.index)] = {
                    'Loc': loc,
                    'Images': f'{INPUT_IMAGES_FOLDER}_{loc}/Image_{imgId}.tif',
                    'Image_crop': f"{OUTPUT_FOLDER}/Images/{loc}_{imgId}_{i}_{j}.jpg",
                    'Mask_crop': f"{OUTPUT_FOLDER}/Masks/{loc}_{imgId}_{i}_{j}.jpg",
                    'Col': j,
                    'Row': i,
                    'X': x,
                    'Y': y,
                    'Asbestos': np.any(crop_mask == 1),
                    'Buildings': np.any(crop_mask == 2),
                    'Streets': np.any(crop_mask == 3),
                    'GreenSpaces': np.any(crop_mask == 4),
                    'Others': np.any(crop_mask == 5)
                }

data_csv.to_csv(f'{OUTPUT_FOLDER}/dataset.csv')
