#%%
import pandas as pd
from utils import apply_lut_mask
from PIL import Image
import cv2
import numpy as np
import os

#%%
dataset = pd.read_csv('dav_dataset/dataset.csv')

#%%
asbestos = dataset[dataset['Class'] == 1]
others = dataset[dataset['Class'] == 2]

#%%
os.makedirs('Dataset_dav_examples', exist_ok=True)
selected = others.sample(10)
for index, row in selected.iterrows():
    
    name = row['Image_crop'].split("/")[-1].split(".")[0]
    img = Image.open(f"{row['Image_crop']}")
    mask = cv2.imread(f"{row['Mask_crop']}", -1).astype(np.uint8)

    mask_rgb = (apply_lut_mask(mask) * 255).astype(np.uint8)
    img.save(f"Dataset_dav_examples/{name}.jpg")
    Image.fromarray(mask_rgb).save(f"Dataset_dav_examples/{name}_mask.jpg")


# %%
