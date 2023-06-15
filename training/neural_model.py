import pythainlp
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

# counter and tokenize
import pandas as pd
import numpy as np
import os
import string
import re
import time
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from collections import Counter
import numpy as np

model_dir = os.path.join(os.getcwd(), "..", "model")

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
    # data.loc[:, 'encoded'] = [encode_sentence(x, vocab2index, max_len) for x in data['comment']]

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
    
def train_model_regr(model, train_loader, test_loader, file_name, index, epochs=10, lr=0.001, early_stopping_measure=float('inf')):
    
    print(f'******************** {file_name} ********************)')
    print(f'train_{index}, test_{index}')
    
    model_path = os.path.join(model_dir, file_name)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    test_time = 0
    
    start_time = time.time()
    for i in range(1, epochs+1):
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
            
        s_time = time.time()
        test_loss = validation_metrics_regr(model, test_loader)
        e_time = time.time()
        t_time = e_time - s_time
        test_time += t_time
        
        if test_loss >= early_stopping_measure:
            print("final epoch:", i, ",train loss %.4f, MAE-test %.4f" % (sum_loss/total, test_loss))
            break
            
        early_stopping_measure = test_loss
            
        if i % 5 == 0:
            print("epoch:", i, ",MAE-train %.4f, MAE-test %.4f" % (sum_loss/total, test_loss))

            
    end_time = time.time()
    train_time = end_time - start_time
    print('train: %.6f, test: %.6f' % (train_time, test_time/epochs))
            
    torch.save(model.state_dict(), model_path)
    
    return model_path
            
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
    
    return (sum_loss/total)

def evaluation_validate(model, val_data, test_loader):
    start_time = time.time()
    model.eval()
    actual = []
    predicted = []
    sum_loss = 0.0
    for x, y, l in test_loader:
        x = x.long()
        y = y.float()
        y_hat = model(x, l)
        loss = F.l1_loss(y_hat, y.unsqueeze(-1))
        actual.extend(y.cpu().numpy().tolist())
        predicted.extend(y_hat.cpu().detach().numpy().tolist())
        sum_loss += loss.item()*y.shape[0]
    mae = sum_loss/len(test_loader.dataset)
    end_time = time.time()
    eval_time = end_time - start_time
    
    # print('validate: %.6f, MAE-val: %.6f' % (eval_time, mae))
    
    result = pd.DataFrame({'comment': val_data['comment'],
                        'actual': actual,
                        'predicted': predicted})
    
    return result

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

class RNN_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        rnn_out, ht = self.rnn(x)
        return self.linear(ht[-1])

class CNN_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.linear1 = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = x.permute(0, 2, 1) # switch to (batch_size, embedding_dim, sequence_length)
        conv_outputs = []
        for conv in self.convs:
            conv_output = conv(x)
            conv_output = F.relu(conv_output)
            max_pool_output = F.max_pool1d(conv_output, conv_output.size()[2])
            conv_outputs.append(max_pool_output.squeeze(-1))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)
    
class DNN_regr(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


