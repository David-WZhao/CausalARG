import torch.utils.data as tud
import pandas as pd
import torch

class ARGDataSet(tud.Dataset):
    def __init__(self, data):
        super(ARGDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.FloatTensor(self.data.iloc[item]['seq_map']), self.data.iloc[item]['type_label'], self.data.iloc[item]['mech_label'], self.data.iloc[item]['anti_label']

class ARGDataLoader(object):
    def __init__(self):
        print("loading data...")
        self.antibiotic_count, self.mechanism_count, self.transfer_count = 15, 6, 2

    def load_test_dataSet(self, batch_size):
        print('loading test data...')
        test_data = pd.read_pickle('./data/test/test.pickle')
        test_data = tud.DataLoader(ARGDataSet(test_data), batch_size=batch_size, shuffle=True, num_workers=0)
        return test_data

    def load_n_cross_data(self, k, batch_size):
        print('loading cross_' + str(k) + ' train_val data ...')
        train_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_train.pickle')
        val_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_val.pickle')
        train_data = tud.DataLoader(ARGDataSet(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
        val_data = tud.DataLoader(ARGDataSet(val_data), batch_size=batch_size, shuffle=True, num_workers=0)
        return train_data, val_data

    def get_data_shape(self):
        return self.transfer_count, self.mechanism_count, self.antibiotic_count

if __name__ == '__main__':
    dataloader = ARGDataLoader()
    dataloader.load_n_cross_data(5, 10)
    # print(len(test_dataloader))
    # index, (seq_map, anti_label, mech_label, type_label) = next(enumerate(train_dataloader))
    # print(seq_map.size())
    # print(anti_label.size())
    # print(mech_label.size())
    # print(type_label.size())
