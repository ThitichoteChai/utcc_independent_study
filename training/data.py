file_name = 'data.xlsx'

# ultility model
import os
import numpy as np
import pandas as pd
import time

# torch component
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# sklearn helper
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# evaluation metric
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold

# pretrain model installer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# feature engineering
def calculate_sum_score(data):
    data['score'] = data['creditability'] + data['content'] + data['expression']
    return data

# split dataset as train_test and validate
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "..", "data")
model_dir = os.path.join(current_dir, "..", "model")

data_path = os.path.join(data_dir, file_name)
print("data_path:", data_path)
print("model_dir:", model_dir)

data = pd.read_excel(data_path)
data = calculate_sum_score(data)
data = data[['comment', 'score']]

data, val_data = train_test_split(data, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_data_splits = []
test_data_splits = []

# Loop through each split, and split the data into train and test dataframes
for i, (train_idx, test_idx) in enumerate(kf.split(data)):
    
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    
    train_data_splits.append(train_data)
    test_data_splits.append(test_data)

    print(f'K={i+1}: train_{i+1}: {len(train_data)} | test_{i+1}: {len(test_data)}')

# Create separate variables for each split's train and test data
train_1, test_1 = train_data_splits[0], test_data_splits[0]
train_2, test_2 = train_data_splits[1], test_data_splits[1]
train_3, test_3 = train_data_splits[2], test_data_splits[2]
train_4, test_4 = train_data_splits[3], test_data_splits[3]
train_5, test_5 = train_data_splits[4], test_data_splits[4]
print(f'Validation: val_data:', len(val_data))

import warnings
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")