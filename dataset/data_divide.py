import torch.utils.data as tud
import random
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json

def save_KFold_data(data, K):
    kf = KFold(n_splits=K)
    cross = 1
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        train_data.to_pickle('data/train_val/cross_' + str(cross) +'_train.pickle')
        val_data.to_pickle('data/train_val/cross_' + str(cross) +'_val.pickle')
        print('cross_' + str(cross) +' train_val data saved...')
        cross += 1

def save_test_data(test_data):
    test_data.to_pickle('./data/test/test.pickle')
    print('test data saved...')

def load_data():
    data = pd.read_pickle('./data/arg_v5_processed.pickle')
    anti_count, mech_count, type_count = 15, 6, 2
    return data, anti_count, mech_count, type_count

def init_data(data, train_rate):
    train_data = data.sample(int(len(data) * train_rate))
    test_data = data.drop(labels=train_data.index)
    return train_data, test_data

if __name__ == '__main__':
    data, anti_count, mech_count, type_count = load_data()
    train_data, test_data = init_data(data, 0.8)
    save_test_data(test_data)
    save_KFold_data(train_data, 5)
