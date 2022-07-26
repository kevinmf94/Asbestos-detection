# %% Imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
import albumentations as A

# %% Paramaters
COLS_CSV = ['Loc', 'Images', 'Image_crop', 'Mask_crop', 'Col', 'Row', 'X', 'Y',
            'Asbestos', 'Streets', 'Buildings', 'GreenSpaces', 'Others']
OUTPUT_FOLDER = "dataset"

# %% Create folders and CSV and generate dataset
data_csv = pd.read_csv(f'{OUTPUT_FOLDER}/dataset.csv')
data_csv_asb = data_csv[data_csv['Asbestos'] == True]
data_aug_csv = pd.DataFrame(columns=COLS_CSV)

transforms = [
    A.Compose([
        A.HorizontalFlip(p=1),
    ]),
    A.Compose([
        A.VerticalFlip(p=1),
    ])
]
"""A.Compose([
        A.RandomSizedCrop(min_max_height=(1, 256), height=256, width=256, p=1),
    ]),
    A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop(min_max_height=(1, 256), height=256, width=256, p=1),
    ]),
    A.Compose([
        A.VerticalFlip(p=1),
        A.RandomSizedCrop(min_max_height=(1, 256), height=256, width=256, p=1),
    ]),"""

try:
    os.mkdir(f"{OUTPUT_FOLDER}/Images_aug")
    os.mkdir(f"{OUTPUT_FOLDER}/Masks_aug")
except:
    pass

print(f"Total asbestos: {data_csv_asb.shape[0]}")
count = 1
for index, row in data_csv_asb.iterrows():
    print(f"Processing {count} of {data_csv_asb.shape[0]}\r", end="")

    name = row['Image_crop'].split("/")[-1].split(".")[0]
    img = Image.open(f"{row['Image_crop']}")
    mask = cv2.imread(f"{row['Mask_crop']}", -1).astype(np.uint8)
    img = np.array(img)

    for i in range(len(transforms)):


        augmented = transforms[i](image=img, mask=mask)
        img_path = f"{OUTPUT_FOLDER}/Images_aug/{name}_aug{i}.bmp"
        mask_path = f"{OUTPUT_FOLDER}/Masks_aug/{name}_aug{i}.bmp"

        Image.fromarray(augmented['image']).save(img_path)
        Image.fromarray(augmented['mask']).save(mask_path)

        data_aug_csv.loc[len(data_csv.index)] = {
                    'Loc': row['Loc'],
                    'Images': row['Images'],
                    'Image_crop': img_path,
                    'Mask_crop': mask_path,
                    'Col': row['Col'],
                    'Row': row['Row'],
                    'X': row['X'],
                    'Y': row['Y'],
                    'Asbestos': row['Asbestos'],
                    'Buildings': row['Buildings'],
                    'Streets': row['Streets'],
                    'GreenSpaces': row['GreenSpaces'],
                    'Others': row['Others']
                }

    count += 1

data_aug_csv.to_csv(f'{OUTPUT_FOLDER}/dataset_aug.csv')
