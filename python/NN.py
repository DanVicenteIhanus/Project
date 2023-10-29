import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler# allow to print the entire width and enitre length from pandas
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.stats import zscore
#allowing for more decimals when printing accuracy
np.set_printoptions(precision=10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#df.loc[df['loudness'] < -100, 'loudness'] = df.drop(df[df['loudness'] < -100].index)['loudness'].mean()
#df.loc[df['energy'] > 500, 'energy'] = df.drop(df[df['energy'] > 500].index)['energy'].mean()


df=pd.read_csv('project_train.csv')
# instead of setting them to mean drop them
df.drop(df[df['tempo'] > 500].index, inplace=True)
df.drop(df[df['loudness'] < -1000].index, inplace=True)
X_old = df.drop('Label', axis=1)
y = df['Label']
scaler = StandardScaler()
X = scaler.fit_transform(X_old)
acc_list = []
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adamax',
                  loss='hinge',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    acc_list.append(val_acc)

print(np.mean(acc_list), np.sqrt(np.var(acc_list)))




