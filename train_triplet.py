import random
import os
import time
from datetime import datetime
from typing import Callable, Optional

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, AveragePrecision, ConfusionMatrix, Precision, \
    Recall, Specificity, PrecisionRecallCurve, MeanMetric
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet18_Weights, resnet18

## Constants
DATASET_FOLDER = 'dav_dataset'
EXPERIMENT = "experiment32"
TRAIN = True
TEST = True

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

        if self.set == 'train':

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
    nn.Linear(512, 1, bias=True),
    nn.Sigmoid()
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

# Weighted sampler
weights = [0.7, 0.3]
samples_weight = np.array([weights[label.int().item()] for item, label in training_set])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=60, sampler=sampler, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=30, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=80, shuffle=False, num_workers=1)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
print('Test set has {} instances'.format(len(test_set)))

# Optimizers specified in the torch.optim package
# loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1, dtype=torch.float32).to(device))
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("LR = lr=0.01")


# Functions for one epoch
def train_one_epoch(epoch_index, tb_writer):
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    i = 0

    tacc = Accuracy().to(device)
    tap = AveragePrecision(pos_label=0).to(device)
    tloss = MeanMetric().to(device)

    for inputs, labels in training_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        labels = labels.int()
        tacc.update(outputs, labels)
        tloss.update(loss.item())
        tap.update(outputs, labels)

        if i % 20 == 19:
            last_loss = tloss.compute()  # loss per batch
            last_acc = tacc.compute()
            last_ap = tap.compute()
            print('  batch {} loss: {} acc {:.4f} AP: {:.4f} '.format(i + 1, last_loss, last_acc, last_ap))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Acc/train', last_acc, tb_x)
            tb_writer.add_scalar('AP/train', last_ap, tb_x)
            tloss.reset()
            tacc.reset()
            tap.reset()

        i += 1

    return last_loss, last_acc, last_ap


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
        avg_loss, avg_acc, avg_ap = train_one_epoch(epoch_number, writer)
        exec_time = time.time() - start_time

        # We don't need gradients on to do reporting
        with torch.no_grad():

            vacc = Accuracy().to(device)
            vap = AveragePrecision(pos_label=0).to(device)
            vloss_avg = MeanMetric().to(device)

            i = 0
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.reshape((labels.shape[0], 1))

                voutputs = model(inputs)
                vloss = loss_fn(voutputs, labels)

                labels = labels.int()
                vacc.update(voutputs, labels)
                vap.update(voutputs, labels)
                vloss_avg.update(vloss.item())
                i += 1

            vloss_avg = vloss_avg.compute()
            print('Time: {} LOSS train {:.4f} Acc {:.4f}% AP: {:.4f} valid {:.4f} Acc {:.4f} AP: {:.4f}'
                  .format(exec_time, avg_loss, avg_acc, avg_ap, vloss_avg, vacc.compute(), vap.compute()))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': vloss_avg},
                               epoch_number + 1)
            writer.add_scalars('Training vs. Validation Acc',
                               {'Training': avg_acc, 'Validation': vacc.compute()},
                               epoch_number + 1)
            writer.add_scalars('Training vs. Validation AP',
                               {'Training': avg_ap, 'Validation': vap.compute()},
                               epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if vloss_avg < best_vloss:
                best_vloss = vloss_avg
                model_path = '{}/model_{}_{}'.format(EXPERIMENT, timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

        epoch_number += 1

    torch.save(model.state_dict(), '{}/model_final'.format(EXPERIMENT))

if TEST:
    weights = torch.load('{}/model_final'.format(EXPERIMENT))
    model.load_state_dict(weights)
    model.to(device)
    model = model.eval()

    with torch.no_grad():

        accuracy = Accuracy().to(device)
        ap_asb = AveragePrecision(pos_label=0).to(device)
        ap_builds = AveragePrecision(pos_label=1).to(device)
        confusion = ConfusionMatrix(num_classes=2).to(device)
        precision = Precision(average='macro', num_classes=2, multiclass=True).to(device)
        recall = Recall(average='macro', num_classes=2, multiclass=True).to(device)
        spe = Specificity().to(device)
        PRC = PrecisionRecallCurve().to(device)

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.int().to(device)
            labels = labels.reshape((labels.shape[0], 1))

            voutputs = model(inputs)

            accuracy.update(voutputs, labels)
            ap_asb.update(voutputs, labels)
            ap_builds.update(voutputs, labels)
            confusion.update(voutputs, labels)
            precision.update(voutputs, labels)
            recall.update(voutputs, labels)
            spe.update(voutputs, labels)
            PRC.update(1 - voutputs, 1 - labels)

        print("---------- Test METRICS -------------")
        print(confusion.compute())
        print("Accuracy {:.4f}".format(accuracy.compute()))
        print("AP Asb {:.4f}".format(ap_asb.compute()))
        print("AP Builds {:.4f}".format(ap_builds.compute()))
        print("Precision {:.4f}".format(precision.compute()))
        print("Recall {:.4f}".format(recall.compute()))
        print("Specifity (Asbestos Rate) {:.4f}".format(spe.compute()))

        precision, recall, thresholds = PRC.compute()

        # create precision recall curve
        fig, ax = plt.subplots()
        ax.plot(recall.cpu(), precision.cpu(), color='purple')

        # add axis labels to plot
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        # display plot
        plt.savefig(f'{EXPERIMENT}/prc.png')



