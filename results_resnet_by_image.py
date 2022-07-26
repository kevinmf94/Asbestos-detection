import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
import matplotlib.pyplot as plt

## Constants
DATASET_FOLDER = 'dav_dataset'
EXPERIMENT = "experiment32"
SET = "test_bages"

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

#print(model)
weights = torch.load('{}/model_final'.format(EXPERIMENT))
model.load_state_dict(weights)
model.to(device)
model = model.eval()

with torch.no_grad():
    data = pd.read_csv(f'{DATASET_FOLDER}/{SET}.csv')
    #asb = data[data['Class'] == 1].reset_index(drop=True)
    del data['Unnamed: 0']
    #del data['Unnamed: 0.1']
    data['Score'] = 0.0

    i = 1
    for index, item in data.iterrows():
        print(f"Processing {i} of {data.shape[0] + 1}\r", end="")
        img = Image.open(data.iloc[index]['Image_crop'])
        img = transforms(img)
        img = torch.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

        img = img.cuda()
        output = model(img)
        data.loc[index, "Score"] = 1 - round(output.item(), 5)
        i += 1

    sort_data = data.sort_values('Score', ascending=False)
    print(sort_data)
    x = np.mgrid[0:sort_data.shape[0]]
    y = sort_data['Score'].to_numpy()

    sort_data.to_csv(f'{EXPERIMENT}/{SET}_results.csv')
    plt.plot(x, y)
    plt.savefig(f"{EXPERIMENT}/{SET}_score_graph.jpg")



