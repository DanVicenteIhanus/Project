import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''
------------------------------------
Handle data
------------------------------------
'''

def import_data(datadir):
    data = pd.read_csv(datadir, index_col=False)
    return data

def parse_data(data: pd.DataFrame):
    features = data.drop(columns='Label')
    labels = data['Label']
    N = len(data.index)
    return features, labels, N

'''
------------------------------------
Support vector machine
------------------------------------
'''

def svm_routine(features, labels, kernel, parameter, coef0, test_size):
    
    # --- split data ----
    (train_features, test_features, 
     train_labels, test_labels) = train_test_split(features, labels, test_size = test_size)
    
    if kernel == 'linear':
        svm_model = SVC(kernel=kernel)
    if kernel == 'poly': 
        svm_model = SVC(kernel=kernel, gamma = parameter, coef0=coef0)
    else:
        svm_model = SVC(kernel=kernel, gamma = parameter)
    
    # --- fit data and predict --- 
    svm_model.fit(train_features, train_labels)
    predicted_labels = svm_model.predict(test_features)

    return test_labels, predicted_labels

'''
------------------------------------
Main script
------------------------------------

'''
if __name__ == '__main__':
    
    # --- Data --- #
    datadir = '/Users/danvicente/Statistik/SF2935 - Modern Methods/Project/data/'
    training_set = 'project_train.csv' # NOTE: project_train.csv has 505 samples and 11 features
    training_data = import_data(datadir+training_set)
    features, labels, N = parse_data(training_data)
    
    # --- Hyperparameters --- #
    gamma = 0.01
    kernel = 'rbf'
    coef0 = 0
    test_size = 0.3 

    # --- Training  ---- #
    test_labels, predicted_labels = svm_routine(features=features,
                                                labels=labels,
                                                kernel = kernel,
                                                parameter = gamma,
                                                coef0=coef0,
                                                test_size=test_size)

    # --- Evaluate SVM --- #
    result = classification_report(test_labels, predicted_labels)
    print(result)