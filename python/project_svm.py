import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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

def clean(data: pd.DataFrame):
    data.loc[data['loudness'] < -100, 'loudness'] = data.drop(data[data['loudness'] < -100].index)['loudness'].mean()
    data.loc[data['energy'] > 500, 'energy'] = data.drop(data[data['energy'] > 500].index)['energy'].mean()
    return data 

'''
------------------------------------
Training-models
------------------------------------
'''

def svm_routine(train_features, train_labels, 
                test_features, test_labels, 
                kernel, parameter, coef0):
    
    # --- kernel choice ---
    if kernel == 'linear':
        svm_model = SVC(kernel=kernel)
    if kernel == 'poly': 
        svm_model = SVC(kernel=kernel, gamma = parameter, coef0 = coef0)
    else:
        svm_model = SVC(kernel=kernel, gamma = parameter)
    
    # --- fit data and predict --- 
    svm_model.fit(train_features, train_labels)
    predicted_labels = svm_model.predict(test_features)

    return test_labels, predicted_labels


def qda_routine(train_features, train_labels, 
                test_features, test_labels):
    clf = QDA()
    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    
    return test_labels, predicted_labels

def compute_accuracy(predicted, true):
    return np.mean(true == predicted)

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
    clean_data = clean(training_data)
    features, labels, N = parse_data(clean_data)
    
    # --- Hyperparameters --- #
    gamma = 0.001
    kernel = 'linear'
    coef0 = 1
    test_size = 0.1 

    # --- split data ---- #
    (train_features, test_features, 
     train_labels, test_labels) = train_test_split(features, labels, test_size = test_size)

    # --- Training SVM ---- #
    test_labels, predicted_labels = svm_routine(train_features = train_features,
                                                train_labels = train_labels,
                                                test_features = test_features,
                                                test_labels = test_labels,
                                                kernel = kernel,
                                                parameter = gamma,
                                                coef0=coef0)
    
    # --- Training QDA --- #
    test_labels_QDA, predicted_labels_QDA = qda_routine(train_features = train_features,
                                                        train_labels = train_labels, 
                                                        test_features = test_features,
                                                        test_labels = test_labels)
    
    # --- Evaluate SVM --- #
    result = classification_report(test_labels, predicted_labels)
    result_qda = classification_report(test_labels_QDA, predicted_labels_QDA)
    
    print(f'--- Classification report from SVM with Kernel= {kernel} ----')
    print(result)
    print(f'Accuracy = {compute_accuracy(predicted_labels, test_labels)}')
    print(f'--- Classification report from QDA ----')
    print(result_qda)
    print(f'Accuracy = {compute_accuracy(predicted_labels_QDA, test_labels_QDA)}')