# %% Imports
import pandas as pd
from sklearn.utils import shuffle

# %% Generate distribution train/val/test
OUTPUT_FOLDER = "dav_dataset_v2"
TRAIN = 0.70
VAL = TRAIN + 0.10
TEST = VAL + 0.20

# Load dataset and seperate with Asbestos/NonAsbestos
data_csv = pd.read_csv(f'{OUTPUT_FOLDER}/dataset.csv')
asbestos = data_csv[data_csv['Class'] == 1]
others = data_csv[data_csv['Class'] == 2]

n_asbestos = asbestos.shape[0]
n_others = others.shape[0]
print(f"Asbestos=[{n_asbestos}] Others=[{n_others}] Total=[{n_asbestos + n_others}]")

# Get random samples from asbestos a divide it
shuffle_asbestos = shuffle(asbestos)
train_asb = shuffle_asbestos.iloc[0:int(n_asbestos * TRAIN)]
val_asb = shuffle_asbestos.iloc[int(n_asbestos * TRAIN): int(n_asbestos * VAL)]
test_asb = shuffle_asbestos.iloc[int(n_asbestos * VAL): int(n_asbestos * TEST)]
print(f'Asbestos -> Train[{train_asb.shape[0]}] Val[{val_asb.shape[0]}] Test[{test_asb.shape[0]}]')

# Get random samples from others and divide it
shuffle_others = shuffle(others)
train_other = shuffle_others.iloc[0:int(n_others * TRAIN)]
val_other = shuffle_others.iloc[int(n_others * TRAIN): int(n_others * VAL)]
test_other = shuffle_others.iloc[int(n_others * VAL): int(n_others * TEST)]
print(f'Others -> Train[{train_other.shape[0]}] Val[{val_other.shape[0]}] Test[{test_other.shape[0]}]')

train = shuffle(pd.concat([train_asb, train_other]))
val = shuffle(pd.concat([val_asb, val_other]))
test = shuffle(pd.concat([test_asb, test_other]))

train.to_csv(f'{OUTPUT_FOLDER}/train.csv')
val.to_csv(f'{OUTPUT_FOLDER}/val.csv')
test.to_csv(f'{OUTPUT_FOLDER}/test.csv')
