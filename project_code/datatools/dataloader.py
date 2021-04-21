import pickle
import torch
import numpy as np

from torch.utils.data import Dataset


class SEEDDataset(Dataset):

    def __init__(self, train, one_hot, test_idx):
        with open('EEG_X.pickle', 'rb') as input_f:
            EEG_X = pickle.load(input_f)
        if one_hot:
            with open('EEG_y_onehot.pickle', 'rb') as input_f:
                EEG_y = pickle.load(input_f)
        else:
            with open('EEG_y.pickle', 'rb') as input_f:
                EEG_y = pickle.load(input_f) + 1
                EEG_y = np.squeeze(EEG_y)
        
        
        if train:
            self.EEG_X = torch.from_numpy(np.concatenate(np.concatenate((EEG_X[:test_idx], EEG_X[test_idx+1:])))).float()
            m = self.EEG_X.mean(dim=0, keepdim=True)
            s = self.EEG_X.std(dim=0, unbiased=False, keepdim=True)
            self.EEG_X = (self.EEG_X - m) / s
            
            if one_hot:
                self.EEG_y = torch.from_numpy(np.concatenate(np.concatenate((EEG_y[:test_idx], EEG_y[test_idx+1:])))).float()
            else:
                self.EEG_y = torch.from_numpy(np.concatenate(np.concatenate((EEG_y[:test_idx], EEG_y[test_idx+1:])))).long()

            self.num_samples = len(self.EEG_X)
        else:
            self.EEG_X = torch.from_numpy(EEG_X[test_idx]).float()
            m = self.EEG_X.mean(dim=0, keepdim=True)
            s = self.EEG_X.std(dim=0, unbiased=False, keepdim=True)
            self.EEG_X = (self.EEG_X - m) / s

            if one_hot:
                self.EEG_y = torch.from_numpy(EEG_y[test_idx]).float()
            else:
                self.EEG_y = torch.from_numpy(EEG_y[test_idx]).long()

            self.num_samples = len(self.EEG_X)
    
    def __getitem__(self, index):
        return self.EEG_X[index], self.EEG_y[index]
    
    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    seed_dataset = SEEDDataset(train=False, one_hot=False)
    inputs, label = seed_dataset[:]
    print(inputs.shape)
    print(label.shape)
