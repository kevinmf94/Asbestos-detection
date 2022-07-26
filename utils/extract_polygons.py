import numpy as np
import cv2
from .preview import apply_lut_mask
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image

ASBESTOS_CLASS = 1


def extract_polygons(mask, value):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))
    binary_mask = (mask == value).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':

    # Check all Masks with asbestos
    data_csv = pd.read_csv(f'../dav_dataset/dataset.csv')
    asbestos = data_csv[data_csv['Class'] == 1]
    for idx, row in asbestos.iterrows():
        im = Image.open(f"../{row['Image_crop']}")
        Masks = cv2.imread(f'../{row["Mask_crop"]}', -1).astype(np.uint8)
        
        polys = extract_polygons(Masks, 1);
        print(row["Mask_crop"])
        print(len(polys))
        #preview_mask_category(Masks, 2)
        
        erode = cv2.morphologyEx(Masks, cv2.MORPH_CLOSE, np.ones((3, 3)))
        mask_lut = apply_lut_mask(erode)
        preview = np.array(im)/255 * 0.6 + mask_lut * 0.4
        plt.imshow(preview)
        plt.show()
        mask_lut = apply_lut_mask(Masks)
        preview = np.array(im)/255 * 0.6 + mask_lut * 0.4
        plt.imshow(preview)
        plt.show()
