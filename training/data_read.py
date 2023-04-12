file_name = '300_data_pop.xlsx'

import os
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "..", "data")
data_path = os.path.join(data_dir, file_name)
model_dir = os.path.join(current_dir, "..", "model")

print("Data path:", data_path)

def calculate_sum_score(data):
    data['score'] = data['creditability'] + data['content'] + data['expression']
    return data
    
def norm_score(data):
    score_min = data['score'].min()
    score_max = data['score'].max()
    data['score_norm'] = (data['score'] - score_min) / (score_max - score_min)
    return data

import pythainlp
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text):
    text = normalize(text)
    tokens = word_tokenize(text, engine='newmm')
    return tokens

import pandas as pd

# Load the data
data = pd.read_excel(data_path)
data = calculate_sum_score(data)
data = norm_score(data)
data = data[['comment', 'score_norm']]

from sklearn.model_selection import train_test_split

# split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print("data_size:", len(data))
print("variable:", 'data', 'train_data', 'test_data')

# import important library

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import numpy as np
