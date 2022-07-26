import random
import os
import time
from datetime import datetime
from typing import Callable, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet18_Weights, resnet18

## Constants
DATASET_FOLDER = 'dav_dataset'
EXPERIMENT = "experiment32"
TRAIN = True
TEST = True
EMBEDDING_DIM = 256

try:
    os.mkdir(EXPERIMENT)
except:
    pass

transform_alb = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(p=0.3),
    A.RandomBrightness(p=0.3),
    A.RandomContrast(p=0.3)
])


class AsbestosDataset(VisionDataset):

    def __init__(self, set: str, folder: str, transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(set, transform=transforms, target_transform=target_transform)
        self.csv = pd.read_csv(f'{folder}/{set}.csv')
        self.set = set

    def __getitem__(self, index: int):

        item = self.csv.iloc[index]
        anchor, target = Image.open(item['Image_crop']), item['Class'] - 1

        if self.set == 'train' or self.set == 'val_bages':

            positive_item = random.choice(self.csv[self.csv['Class'] == target])
            positive_img = Image.open(positive_item['Image_crop'])

            negative_item = random.choice(self.csv[self.csv['Class'] != target])
            negative_img = Image.open(negative_item['Image_crop'])

            if target == 0 and self.set == 'train':
                positive_img = transform_alb(image=np.array(positive_img))['image']
                positive_img = Image.fromarray(positive_img)

            if self.transform is not None:
                anchor = self.transform(anchor)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor, positive_img, negative_img, torch.tensor(target, dtype=torch.float32)

        else:

            if self.transform is not None:
                anchor = self.transform(anchor)

            return anchor, torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]


# Check CUDA
print("CUDA ON: " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Model
weights = ResNet18_Weights.IMAGENET1K_V1
transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
model = resnet18(weights=weights)
model.fc = nn.Sequential(
    nn.Linear(512, EMBEDDING_DIM, bias=True)
)

model.to(device)
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print("Learnable: {} ".format(sum([np.prod(p.size()) for p in model_parameters])))
print("Total Parameters: {}".format(sum([np.prod(p.size()) for p in model.parameters()])))

# Create datasets for training & validation, download if necessary
training_set = AsbestosDataset('train', DATASET_FOLDER, transforms=transforms)
validation_set = AsbestosDataset('val_bages', DATASET_FOLDER, transforms=transforms)
test_set = AsbestosDataset('test_bages', DATASET_FOLDER, transforms=transforms)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=60, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=30, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=False, num_workers=1)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
print('Test set has {} instances'.format(len(test_set)))

# Optimizers specified in the torch.optim package
# loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1, dtype=torch.float32).to(device))
loss_fn = torch.nn.TripletMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("LR = lr=0.01")


# Functions for one epoch
def train_one_epoch(epoch_index, tb_writer):
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    i = 0
    tloss = MeanMetric().to(device)

    for anchor_img, positive_img, negative_img, anchor_label in training_loader:

        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        # Compute the loss and its gradients
        loss = loss_fn(anchor_out, positive_out, negative_out, anchor_label)
        loss.backward()
        optimizer.step()

        tloss.update(loss.item())

        if i % 20 == 19:
            last_loss = tloss.compute()  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tloss.reset()

        i += 1

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
if TRAIN:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('{}/resnet_trainer_{}'.format(EXPERIMENT, timestamp))

    epoch_number = 0
    EPOCHS = 50
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        start_time = time.time()
        avg_loss = train_one_epoch(epoch_number, writer)
        exec_time = time.time() - start_time

        # We don't need gradients on to do reporting
        with torch.no_grad():

            vloss_avg = MeanMetric().to(device)

            i = 0
            for  anchor_img, positive_img, negative_img, anchor_label in validation_loader:

                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                negative_img = negative_img.to(device)

                # Make predictions for this batch
                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)

                vloss = loss_fn(anchor_out, positive_out, negative_out)
                vloss_avg.update(vloss.item())

                i += 1

            vloss_avg = vloss_avg.compute()
            print('Time: {} LOSS train {:.4f} LOSS valid {:.4f}'
                  .format(exec_time, avg_loss, vloss_avg))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': vloss_avg},
                               epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if vloss_avg < best_vloss:
                best_vloss = vloss_avg
                model_path = '{}/model_{}_{}'.format(EXPERIMENT, timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

        epoch_number += 1

    torch.save(model.state_dict(), '{}/model_final'.format(EXPERIMENT))



