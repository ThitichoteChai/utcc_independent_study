import pythainlp
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

# counter and tokenize
import pandas as pd
import numpy as np
import string
import re
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from collections import Counter
import numpy as np

def thai_tokenizer(text):
    text = normalize(text)
    tokens = word_tokenize(text, engine='newmm')
    return tokens

def preprocess_data(data, max_len=100, min_freq=2, vocab2index=None):
    
    # Tokenize and count word frequency
    counts = Counter()
    for text in list(data['comment']):
        counts.update(thai_tokenizer(text))

    # Filter by word frequency
    counts = {word: freq for word, freq in counts.items() if freq >= min_freq}

    # Create vocabulary
    vocab2index = {word: i for i, word in enumerate(['', 'UNK'] + list(counts.keys()))}

    # Encode text data into integer sequences
    def encode_sentence(text, vocab2index, max_len):
        tokenized = thai_tokenizer(text)
        encoded = np.array([vocab2index.get(word, 1) for word in tokenized])
        return encoded[:max_len] if len(encoded) > max_len else np.pad(encoded, (0, max_len - len(encoded))), 

    data.loc[:, 'encoded'] = data['comment'].apply(lambda x: encode_sentence(x, vocab2index, max_len)).tolist()
    data = data[['encoded', 'score']]
    
    return data, vocab2index

class ThaiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # print(self.X[idx][0])
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx]
    
def train_model_regr(model, train_loader, test_loader, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        # print('epochs:', i, end=' ')
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_loader:
            x = x.long()
            y = y.float()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.l1_loss(y_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        test_loss = validation_metrics_regr(model, test_loader)
        if i % 5 == 1:
            print("train mae %.4f test mae %.4f" % (sum_loss/total, test_loss))
            
def validation_metrics_regr(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y, l in test_loader:
        x = x.long()
        y = y.float()
        y_hat = model(x, l)
        loss = F.l1_loss(y_hat, y.unsqueeze(-1))
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total

class LSTM_regr(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

    
