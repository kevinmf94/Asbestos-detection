import pickle
import pandas as pd
import cv2
from detectron2.structures import BoxMode
from utils import extract_polygons
import pycocotools as coco
import numpy as np
from sklearn.utils import shuffle

INSTANCES_TRAIN = True
DATA_AUG = True
# Classes: 1 (Asbestos), 2 (Buildings), 3 (Streets), 4 (Greenspaces), 5 (Others)
#CLASSES_TO_GENERATE = [1, 2, 3, 4]
TH_SIZE_COMPONENT = 1000
#CLASSES_TO_GENERATE = [2, 3, 4]
CLASSES_TO_GENERATE = [1, 2]


def get_dict(csv):

    dataset_dicts = []
    count = 1
    for index, row in csv.iterrows():
        print(f"    \rImage {count} of {csv.shape[0]}", end=' ')
        record = {}
        mask = cv2.imread(f'{row["Mask_crop"]}', -1).astype(np.uint8)
        height, width = mask.shape[0:2]

        record["file_name"] = row['Image_crop']
        record["file_name_mask"] = row['Mask_crop']
        record["image_id"] = row[0]
        record["height"] = height
        record["width"] = width
        record["Asbestos"] = row['Asbestos']

        objs = []

        ids = np.unique(mask)

        # Generate objects for each class defined in the config
        for class_id in CLASSES_TO_GENERATE:
            
            if class_id in ids:
                
                mask_f = (mask == class_id).astype(np.uint8)
                result = cv2.connectedComponentsWithStats(mask_f)
                
                for i in range(1, result[0]):
                    x, y, w, h, size = result[2][i]
                    
                    if size > TH_SIZE_COMPONENT:
                        encode = coco.mask.encode(np.asarray(result[1] == i, order="F"))
                        objs.append({
                            "bbox": coco.mask.toBbox(encode),
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": encode,
                            "category_id": class_id - 1
                        })
                   

        record["annotations"] = objs
        dataset_dicts.append(record)
        count += 1

    return dataset_dicts


if __name__ == '__main__':

    GENERATE = True

    if GENERATE:
        print("Generating dictionaries...")
        with open('../dataset/train_dec.pkl', 'wb') as file:
            csv = pd.read_csv(f'../dataset/train.csv')
            if DATA_AUG:
                csv_aug = pd.read_csv(f'../dataset/dataset_aug.csv')
                csv = shuffle(pd.concat([csv, csv_aug]))

            pickle.dump(get_dict(csv), file)
            print(" Train generated!")

        with open(f'../dataset/val_dec.pkl', 'wb') as file:
            csv = pd.read_csv(f'../dataset/val.csv')
            pickle.dump(get_dict(csv), file)
            print(" Val generated!")

        with open(f'../dataset/test_dec.pkl', 'wb') as file:
            csv = pd.read_csv(f'../dataset/test.csv')
            pickle.dump(get_dict(csv), file)
            print(" Test generated!")

