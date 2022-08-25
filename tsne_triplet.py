from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet18_Weights, resnet18
import seaborn as sns

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
test_set = AsbestosDataset('test_bages', DATASET_FOLDER, transforms=transforms)

# Create data loaders for our datasets; shuffle for training, not for validation
test_loader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=False, num_workers=1)

with torch.no_grad():
    x = []
    y = []
    for anchor, label in test_loader:
        anchor = anchor.to(device)
        feature = model(anchor).cpu()
        x.extend(feature.numpy())
        y.extend(label.int().numpy())
        print(len(y))

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    for i in [5, 10, 15, 30, 50, 80, 100]:
        tsne = TSNE(n_components=2, verbose=1, perplexity=i, n_iter=20000)
        result = tsne.fit_transform(x, y)

        """plt.figure(figsize=(16, 10))
        plt.scatter(result[:,0], result[:, 1], data=y)
        plt.savefig(f"{EXPERIMENT}/tsne.png")"""
        plot = sns.scatterplot(x=result[:,0], y=result[:, 1], hue=y,
                        palette=sns.color_palette("hls", 2))
        plt.savefig(f"{EXPERIMENT}/tsne_{i}_20000.png")
        plt.clf()
