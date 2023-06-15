# tradition model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import joblib
import time
import os

import pythainlp
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text):
    text = normalize(text)
    tokens = word_tokenize(text, engine='newmm')
    return tokens

vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer)

current_dir = os.getcwd()
model_dir = os.path.join(current_dir, "..", "model")

def evaluate_model(file_name, val_data):
    # print(f'******************** {file_name} ********************)')
    model = joblib.load(os.path.join(model_dir, file_name))
    
    X_validate = vectorizer.transform(val_data['comment'])
    y_actual = val_data['score']
    
    start_time = time.time()
    y_pred = model.predict(X_validate)
    y_pred = np.clip(y_pred, 0, 9)
    end_time = time.time()
    validate_time = end_time - start_time
    
    mae = mean_absolute_error(y_actual, y_pred)
    
    print(f'{file_name}: validation mae: {mae:.4f}')
    
    result = pd.DataFrame({'comment': val_data['comment'],
                            'actual': y_actual,
                            'predicted': y_pred})

    result = result.sort_values(by='predicted', ascending=False)
    print('validation: %.6f' % validate_time)
    
    return result

def linear_model(X_train, X_test, y_train, y_test, index):
    
    file_name = 'linear_model_' + str(index) + '.pkl'
    
    print(f'******************** {file_name} ********************)')
    print(f'train_{index}, test_{index}')
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    param_grid_linear = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
    
    grid_search_linear = GridSearchCV(
        LinearRegression(),
        param_grid=param_grid_linear,
        n_jobs = -1,
        verbose = 1
    )
    
    # training
    start_time = time.time()
    grid_search_linear.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    mae = grid_search_linear.best_score_
    print(f'Linear Regression MAE on train set: {mae:.4f}')
    
    # testing
    start_time = time.time()
    y_pred = grid_search_linear.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Linear Regression MAE on test set: {mae:.4f}')
    
    best_params = grid_search_linear.best_params_
    
    print('train: %.6f' % train_time)
    print('test: %.6f' % test_time)
    
    # save model
    joblib.dump(grid_search_linear, os.path.join(model_dir, file_name))

def svm_model(X_train, X_test, y_train, y_test, index):
    
    file_name = 'svm_model_' + str(index) + '.pkl'
    
    print(f'******************** {file_name} ********************)')
    print(f'train_{index}, test_{index}')
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    param_grid_svm = {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.3],
        'gamma': ['scale', 'auto'],
    }
    
    grid_search_svm = GridSearchCV(
        SVR(),
        param_grid=param_grid_svm,
        n_jobs = -1,
        verbose = 1
    )
    
    # training
    start_time = time.time()
    grid_search_svm.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    mae = grid_search_svm.best_score_
    print(f'SVM MAE on train set: {mae:.4f}')
    
    # testing
    start_time = time.time()
    y_pred = grid_search_svm.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'SVM MAE on test set: {mae:.4f}')
    
    best_params = grid_search_svm.best_params_
    
    print('train: %.6f' % train_time)
    print('test: %.6f' % test_time)
    
    # save model
    joblib.dump(grid_search_svm, os.path.join(model_dir, file_name))
    
def tree_model(X_train, X_test, y_train, y_test, index):
    
    file_name = 'tree_model_' + str(index) + '.pkl'
    
    print(f'******************** {file_name} ********************)')
    print(f'train_{index}, test_{index}')
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    param_grid_tree = {
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search_tree = GridSearchCV(
        DecisionTreeRegressor(),
        param_grid=param_grid_tree,
        n_jobs = -1,
        verbose = 1
    )
    
    # training
    start_time = time.time()
    grid_search_tree.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    mae = grid_search_tree.best_score_
    print(f'Decision Tree MAE on train set: {mae:.4f}')
    
    # testing
    start_time = time.time()
    y_pred = grid_search_tree.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Decision Tree MAE on test set: {mae:.4f}')
    
    best_params = grid_search_tree.best_params_
    
    print('train: %.6f' % train_time)
    print('test: %.6f' % test_time)
    
    # save model
    joblib.dump(grid_search_tree, os.path.join(model_dir, file_name))
    
def knn_model(X_train, X_test, y_train, y_test, index):
    
    file_name = 'knn_model_' + str(index) + '.pkl'
    
    print(f'******************** {file_name} ********************')
    print(f'train_{index}, test_{index}')
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    param_grid_knn = {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'leaf_size': [10, 20, 30]
    }
    
    grid_search_knn = GridSearchCV(
        KNeighborsRegressor(),
        param_grid=param_grid_knn,
        n_jobs = -1,
        verbose = 1
    )
    
    # training
    start_time = time.time()
    grid_search_knn.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    mae = grid_search_knn.best_score_
    print(f'KNN MAE on train set: {mae:.4f}')
    
    # testing
    start_time = time.time()
    y_pred = grid_search_knn.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'KNN MAE on test set: {mae:.4f}')
    
    best_params = grid_search_knn.best_params_
    
    print('train: %.6f' % train_time)
    print('test: %.6f' % test_time)
    
    # save model
    joblib.dump(grid_search_knn, os.path.join(model_dir, file_name))