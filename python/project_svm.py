import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
import sklearn as sk


def import_data(datadir):
    data = pd.read_csv(datadir)
    return data

def main():
    datadir='...'
    data = import_data(datadir)
    