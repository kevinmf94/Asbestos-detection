# %% Imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
from utils import postprocess_and_save, postprocessmask_and_save
from sklearn.utils import shuffle

# %% Paramaters
COLS_CSV = ['Loc', 'Images', 'Image_crop', 'Mask_crop', 'Class', 'RealClass', 'Aug']
RESIZE_ENABLED = True
RESIZE = (256, 256)
IMAGE_SETS = ['Bages', 'Zona_Franca']
N_IMAGES_SETS = {
    'Bages': 62,
    'Zona_Franca': 5
}
INPUT_IMAGES_FOLDER = "images"
OUTPUT_FOLDER = "dav_dataset"
OTHERS_CLASS = 4
TH_SIZE = 1000
ASBESTOS = 1
HARD_NEG = 2
NEG = 3

CLASSES = {1: 1, 2: 2, 3: 2}

# %% Create folders and CSV and generate dataset
os.makedirs(f"{OUTPUT_FOLDER}/Test_images", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/Test_masks", exist_ok=True)

data_csv = pd.DataFrame(columns=COLS_CSV)

for loc in IMAGE_SETS:
    print(f"Processing {loc}, n_images = {N_IMAGES_SETS[loc]}")

    for imgId in range(1, N_IMAGES_SETS[loc] + 1):
        print(f"    Processing Images {imgId} of {N_IMAGES_SETS[loc]}\r", end="")
        img = Image.open(f"{INPUT_IMAGES_FOLDER}/{loc}/Images/image_{imgId}.tif")
        mask = cv2.imread(f"{INPUT_IMAGES_FOLDER}/{loc}/Test_Masks/image_{imgId}_test.tif", -1).astype(np.uint8)
        img = np.array(img)

        for classId in CLASSES.keys():
            conn = cv2.connectedComponentsWithStats((mask == classId).astype(np.uint8))
            for i in range(1, conn[0]):
                x, y, w, h, size = conn[2][i]

                if size > TH_SIZE:
                    crop_img = img[y:y + h, x:x + w]
                    crop_mask = mask[y:y + h, x:x + w]
                    #crop_mask = ((conn[1] == i)[y:y + h, x:x + w] * classId).astype(np.uint8)
                    postprocess_and_save(crop_img, f"{OUTPUT_FOLDER}/Test_images/{loc}_{imgId}_{i}_{classId}.png")

                    # Transforms mask to binary problem
                    crop_mask[crop_mask == 3] = 2
                    postprocessmask_and_save(crop_mask, f"{OUTPUT_FOLDER}/Test_masks/{loc}_{imgId}_{i}_{classId}.png", other_class=OTHERS_CLASS)

                    data_csv.loc[len(data_csv.index)] = {
                        'Loc': loc,
                        'Images': f'{INPUT_IMAGES_FOLDER}/{loc}/Image_{imgId}.tif',
                        'Image_crop': f"{OUTPUT_FOLDER}/Test_images/{loc}_{imgId}_{i}_{classId}.png",
                        'Mask_crop': f"{OUTPUT_FOLDER}/Test_masks/{loc}_{imgId}_{i}_{classId}.png",
                        'Class': CLASSES[classId],
                        'RealClass': classId,
                        'Aug': 0
                    }

# Split test dataset in Validation and Training
asbestos = data_csv[data_csv['Class'] == 1]
others = data_csv[data_csv['Class'] == 2]

n_asbestos = asbestos.shape[0]
n_others = others.shape[0]

# Get random samples from asbestos a divide it
val_asb = asbestos.iloc[0: int(n_asbestos * 0.2)]
test_asb = asbestos.iloc[int(n_asbestos * 0.2): n_asbestos]
print(f'Asbestos -> Val[{val_asb.shape[0]}] Test[{test_asb.shape[0]}]')

# Get random samples from others and divide it
val_other = others.iloc[0: int(n_others * 0.2)]
test_other = others.iloc[int(n_others * 0.2): n_others]
print(f'Others -> Val[{val_other.shape[0]}] Test[{test_other.shape[0]}]')

val = shuffle(pd.concat([val_asb, val_other]))
test = shuffle(pd.concat([test_asb, test_other]))

val.to_csv(f'{OUTPUT_FOLDER}/val_set.csv')
test.to_csv(f'{OUTPUT_FOLDER}/test_set.csv')
