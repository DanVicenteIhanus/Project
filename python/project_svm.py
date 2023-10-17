import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from scipy import stats

'''
------------------------------------
Handle data
------------------------------------
'''

def import_data(datadir):
    return pd.read_csv(datadir, index_col=False)

def parse_data(data: pd.DataFrame):
    features = data.drop(columns='Label')
    labels = data['Label']
    N = len(data.index)
    return features, labels, N

def clean(data: pd.DataFrame):
    data.drop_duplicates(inplace=True)
    data.loc[data['loudness'] < -100, 'loudness'] = data.drop(data[data['loudness'] < -100].index)['loudness'].mean()
    data.loc[data['energy'] > 500, 'energy'] = data.drop(data[data['energy'] > 500].index)['energy'].mean()
    return data 

def compute_PCA(data):
    std_scaler = StandardScaler()
    scaled_data = std_scaler.fit_transform(data)
    
    # find PCA dimension using the MLE
    pca = PCA(n_components = 8, svd_solver='auto')
    pca_data = pca.fit_transform(scaled_data)
    
    return pca_data

def remove_outliers(data: pd.DataFrame, ZSCORE_THREASHOLD: int = 4) -> pd.DataFrame:
    zscore = np.abs(stats.zscore(data.select_dtypes(include=["float", "int"])))
    is_inlier = ~ (zscore > ZSCORE_THREASHOLD).any(axis=1)
    data = data[is_inlier]
    return data


'''
------------------------------------
Training-models
------------------------------------
'''

def svm_routine(train_features, train_labels, 
                test_features, 
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

    return predicted_labels


def qda_routine(train_features, train_labels, 
                test_features):
    clf = QDA()
    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    
    return predicted_labels

def lda_routine(train_features, train_labels, 
                test_features):
    clf = LDA()
    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    return predicted_labels

def log_regression(train_features, train_labels, 
                test_features):
    log_model = LR()
    log_model.fit(train_features, train_labels)
    predicted_labels = log_model.predict(test_features)
    return predicted_labels

def compute_accuracy(predicted, true):
    return np.mean(true == predicted)

def manual_accuracy(skf, features, labels, kernel, gamma, coef0):
    # --- Manual calculation of mean accuracies --- #
    lda_acc = []
    svm_acc = []
    for train_ind, test_ind in skf.split(features, labels):
        train_features = features.iloc[train_ind]
        test_features = features.iloc[test_ind]
        train_labels = labels.iloc[train_ind]
        test_labels = labels.iloc[test_ind]


        # --- Training SVM ---- #
        predicted_labels = svm_routine(train_features = train_features,
                                                    train_labels = train_labels,
                                                    test_features = test_features,
                                                    kernel = kernel,
                                                    parameter = gamma,
                                                    coef0=coef0)
        
        # --- Training QDA --- #
        predicted_labels_QDA = qda_routine(train_features = train_features,
                                                            train_labels = train_labels, 
                                                            test_features = test_features)
        
        # --- Training LDA --- #
        predicted_labels_LDA = lda_routine(train_features = train_features,
                                                            train_labels = train_labels, 
                                                            test_features = test_features)
        
        # --- Evaluate models --- #
        result = classification_report(test_labels, predicted_labels)
        result_qda = classification_report(test_labels, predicted_labels_QDA)
        result_lda = classification_report(test_labels, predicted_labels_LDA)

        #print(f'--- Classification report from SVM with Kernel= {kernel} ----')
        #print(result)
        #print(f'Accuracy = {compute_accuracy(predicted_labels, test_labels)}')
        #print(f'--- Classification report from QDA ----')
        #print(result_qda)
        #print(f'Accuracy = {compute_accuracy(predicted_labels_QDA, test_labels)}')
        #print(f'--- Classification report from LDA ----')
        #print(result_lda)
        #print(f'Accuracy = {compute_accuracy(predicted_labels_LDA, test_labels)}')
        #print('----')
        lda_acc.append(compute_accuracy(predicted_labels_LDA, test_labels))
        svm_acc.append(compute_accuracy(predicted_labels, test_labels))
    print(f'LDA acc = {np.mean(lda_acc)}')
    print(f'SVM acc = {np.mean(svm_acc)}')
    return

'''
------------------------------------
Main script
------------------------------------
'''

if __name__ == '__main__':
    
    # --- Data --- #
    datadir = '/Users/danvicente/Statistik/SF2935 - Modern Methods/Project/data/'
    training_set = 'project_train.csv' # NOTE: project_train.csv has 505 samples and 11 features
    test_set ='project_test.csv'
    training_data = import_data(datadir+training_set)
    test_data = import_data(datadir+test_set)

    clean_data = clean(training_data)
    features, labels, N = parse_data(clean_data)

    # normalize data
    std = StandardScaler()
    norm_features = std.fit_transform(features)

    # --- Hyperparameters --- #
    gamma = 0.22778
    kernel = 'rbf'
    coef0 = 1
    test_size = 0.7

    # --- split data ---- #
    #(train_features, test_features, train_labels, test_labels) = train_test_split(features, labels, test_size = test_size)
    
    # --- Compute PCA --- #
    pca_features = compute_PCA(features)
    #pca_labels = compute_PCA(labels)


    # --- Check cross validation of model --- #
    svc_norm = []
    bagged_lda_norm = []
    svc_pca = []
    bagged_lda_pca = []

    for k in range(2):

        # --- PCA data --- #
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        cv_pca_svc = cross_val_score(SVC(kernel=kernel, gamma=gamma), pca_features, labels, scoring='accuracy',
                        cv = skf, verbose=0)
        cv_pca_lda = cross_val_score(LDA(), pca_features, labels, 
                                scoring = 'accuracy', cv = skf, verbose=0)
        
        # --- Normalized  data --- #
        skf2 = StratifiedKFold(n_splits=10, shuffle=True)
        cv_norm_svc = cross_val_score(SVC(kernel=kernel, gamma=0.1), norm_features, labels, scoring='accuracy',
                        cv = skf2, verbose=0)
        cv_norm_lda = cross_val_score(LDA(), norm_features, labels, 
                                scoring = 'accuracy', cv = skf2, verbose=0)
        
        #print(f'cross val score LDA (PCA): {np.mean(cv_pca_lda)}')
        #print(f'cross val score SVM (PCA): {np.mean(cv_pca_svc)}')
        #print(f'cross val score LDA (normalized data) : {np.mean(cv_norm_lda)}')
        #print(f'cross val score SVM (normalized data) : {np.mean(cv_norm_svc)}')
        svc_norm.append(cv_norm_svc)
        svc_pca.append(cv_pca_svc)
        bagged_lda_norm.append(cv_norm_lda)
        bagged_lda_pca.append(cv_pca_lda)
    
    print('======================= SVM ==========================')
    print(f'Mean of accuracy (normalized data) = {np.mean(svc_norm)}')
    print(f'Mean of accuracy (PCA) = {np.mean(svc_pca)}')
    print('======================= LDA ==========================')
    print(f'Mean of accuracy (normalized data) = {np.mean(bagged_lda_norm)}')
    print(f'Mean of accuracy (PCA) = {np.mean(bagged_lda_pca)}')

    # --- train model on full data --- #
    
    model = SVC(kernel='rbf', gamma=0.1)
    model2 = LDA()
    model3 = GaussianNB()

    model.fit(norm_features, labels)
    model2.fit(norm_features, labels)
    model3.fit(norm_features, labels)

    std = StandardScaler()
    norm_data = std.fit_transform(test_data)
    classes = model.predict(norm_data)
    classes2 = model2.predict(norm_data)
    classes3 = model3.predict(norm_data)
    print(classes)
    '''
    [0, 0, 1, 0, 1, 1, 0, 1,
      1, 1, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 1,
              1, 0, 0, 1, 0, 0, 0, 0,
                0, 1, 1, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 1,
                    0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]
    '''
    print('===================== c.f RF =======================')
    anton_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
             0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
               1, 1, 0, 1, 1, 0, 1, 1, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1,
                       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    
    anton_2 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                    1, 1, 0, 1, 1, 0, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 1, 0, 1,
                        1, 1, 1, 0, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 0, 1, 1, 1,
                            0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    
    anton_3 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                    1, 1, 0, 1, 1, 0, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 1, 0, 1,
                        1, 1, 1, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 1, 1,
                            1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    print(f'Length of test-set: {len(anton_1)}')
    print(f'Amount of common classifications between RF and SVM: {len(np.where(classes==anton_3)[0])}')

    print('==================== c.f KNN ========================')
    marwin_preds = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1,
                     0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 
                     0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 1, 0, 1, 1, 0]
    print(f'Amount of common classifications between KNN and SVM: {len(np.where(classes==marwin_preds)[0])}')
    
    print('================= c.f other models =====================')
    print(f'Amount of common classifications between LDA and SVM: {len(np.where(classes2 == classes)[0])}')
    print(f'Amount of common classifications between NB and SVM: {len(np.where(classes3 == classes)[0])}')
    print(f'Amount of common classifications between RF and NB: {len(np.where(classes3 == anton_3)[0])}')
    print(f'Amount of common classifications between RF and LDA: {len(np.where(classes2 == anton_3)[0])}')