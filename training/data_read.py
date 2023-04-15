#### import important library###################################################

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

##### import important library###################################################

import numpy as np
import pandas as pd
import re
from collections import Counter
import string
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

##### import important library###################################################

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

################################# data directory ################################

file_name = '300_data_pop.xlsx'

import os

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "..", "data")
data_path = os.path.join(data_dir, file_name)
model_dir = os.path.join(current_dir, "..", "model")

print("Data path:", data_path)

########################### feature engineer #####################################

import pandas as pd
import numpy as np

def calculate_sum_score(data):
    data['score'] = data['creditability'] + data['content'] + data['expression']
    return data
    
def norm_score(data):
    score_min = data['score'].min()
    score_max = data['score'].max()
    data['score_norm'] = (data['score'] - score_min) / (score_max - score_min)
    return data

# Load the data
data = pd.read_excel(data_path)
data = calculate_sum_score(data)
data = norm_score(data)
data = data[['comment', 'score_norm']]

#################################################################################

import pythainlp
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text):
    text = normalize(text)
    tokens = word_tokenize(text, engine='newmm')
    return tokens

def preprocess_data(data, max_len=200, min_freq=2):

    counts = Counter()
    for text in list(data['comment']):
        counts.update(thai_tokenizer(text))

    print("num_words before:", len(counts.keys()))
    for word in list(counts):
        if counts[word] < min_freq:
            del counts[word]
    print("num_words after:", len(counts.keys()))

    # Creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    def encode_sentence(text, vocab2index, N=max_len):
        tokenized = thai_tokenizer(text)
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded

    data['encoded'] = data['comment'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))
    
    return data, vocab2index

#################################################################################

def create_datasets(X, Y, vocab2index, batch_size):
    class CommonLitReadabiltyDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.y = Y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx].astype(np.int32)), self.y[idx], self.X[idx][1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_ds = CommonLitReadabiltyDataset(X_train, y_train)
    valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid)

    vocab_size = len(vocab2index)
    embedding_dim = 300
    hidden_dim = 200

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    return train_dl, val_dl, vocab_size, embedding_dim, hidden_dim

#################################################################################

from sklearn.model_selection import train_test_split

# split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print("data_size:", len(data))
print("variable:", 'data', 'train_data', 'test_data')

#################################################################################
