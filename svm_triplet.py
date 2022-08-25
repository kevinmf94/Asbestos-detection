from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet18_Weights, resnet18
from sklearn import svm
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

## Constants
DATASET_FOLDER = 'dav_dataset'
EXPERIMENT = "triplet2"
TRAIN = True
TEST = True
EMBEDDING_DIM = 128

class_tranform = {2: 0, 1: 1}


class AsbestosDataset(VisionDataset):

    def __init__(self, set: str, folder: str, transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(set, transform=transforms, target_transform=target_transform)
        self.csv = pd.read_csv(f'{folder}/{set}.csv')
        self.set = set

    def __getitem__(self, index: int):
        item = self.csv.iloc[index]
        anchor, target, csv_class = Image.open(item['Image_crop']), class_tranform[item['Class']], item['Class']

        if self.transform is not None:
            anchor = self.transform(anchor)

        return anchor, torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.csv.shape[0]


# Check CUDA
print("CUDA ON: " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Model
transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
model = resnet18()
model.fc = nn.Sequential(
    nn.Linear(512, EMBEDDING_DIM, bias=True)
)

weights = torch.load('{}/model_final'.format(EXPERIMENT))
model.load_state_dict(weights)
model.to(device)
model = model.eval()

model.to(device)

# Create datasets for training & validation, download if necessary
train_set = AsbestosDataset('train', DATASET_FOLDER, transforms=transforms)
test_set = AsbestosDataset('test_bages', DATASET_FOLDER, transforms=transforms)

# Create data loaders for our datasets; shuffle for training, not for validation
train_loader = torch.utils.data.DataLoader(train_set, batch_size=80, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=False, num_workers=1)

with torch.no_grad():
    x_train = []
    y_train = []

    print("Train set to features")
    for anchor, label in train_loader:
        anchor = anchor.to(device)
        feature = model(anchor).cpu()
        x_train.extend(feature.numpy())
        y_train.extend(label.int().numpy())
        print(len(x_train))

    for i in [4]:
        print("Train SVM Poly " + str(i))
        clf = svm.SVC(kernel="poly", C=0.75, class_weight={0: 0.1, 1: 2}, degree=i)
        clf.fit(x_train, y_train)

        print("Generate output")
        y_pred = []
        y_gt = []
        for anchor, label in test_loader:
            anchor = anchor.to(device)
            feature = model(anchor).cpu()

            output = clf.predict(feature.numpy())
            y_pred.extend(output)
            y_gt.extend(label.int().numpy())

        print("Results " + str(i))
        print(confusion_matrix(y_gt, y_pred))
        print(recall_score(y_gt, y_pred))
        print(accuracy_score(y_gt, y_pred))
