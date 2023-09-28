import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
import sklearn as sk
import os

def import_data(datadir):
    data = pd.read_csv(datadir)
    return data

def main():
    datadir = '/Users/danvicente/Statistik/SF2935\ -\ Modern\ Methods/Project/data'
    file = 'project_train.csv'
    data = import_data(datadir+file)
    print(data)

main()