# Example reading Images/Masks
#%%
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from preview import preview_mask, preview_img_and_mask, apply_lut_mask

#%% Read Images
IMG_ID = 51
LOC = 'Bages'
im = Image.open(f"../images/{LOC}/Images/image_{IMG_ID}.tif")

print(f"Shape {im.size}")
plt.imshow(np.array(im))
plt.show()

#%% Read Masks
mask = cv2.imread(f"../images/{LOC}/Test_Masks/Image_{IMG_ID}_test.tif", -1).astype(np.uint8)
print(np.unique(mask))
mask[mask == 127] = 5
mask[mask == 0] = 5
print(np.unique(mask))
print(f"Classes {np.unique(mask)}")
print(f"Shape {mask.shape}")

# Classes: 1 (Asbestos), 2 (Buildings), 3 (Streets), 4 (Greenspaces), 5 (Others)
plt.imshow(mask, cmap='gray')
plt.show()

#%%
mask_lut = apply_lut_mask(mask)
preview = np.array(im)/255 * 0.6 + mask_lut * 0.4
plt.imshow(preview[800:1600, 800:1600])
plt.show()


#%%
preview_img_and_mask(np.array(im)[300:700, 300:700], mask[300:700, 300:700])

#%% Preview Mask LUT
preview_mask(mask)

# %%
import rasterio
imageio = rasterio.open(f"images/Montornes_del_Valles/Images/Image_2.tif")