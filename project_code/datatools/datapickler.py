import numpy as np
import pickle

from scipy.io import loadmat


EEG_X = loadmat('../../SEED-III/EEG_X.mat')['X'][0]
EEG_y = loadmat('../../SEED-III/EEG_Y.mat')['Y'][0]

EEG_X = np.stack(EEG_X)
EEG_y = np.stack(EEG_y)


with open('EEG_X.pickle', 'wb') as output_f:
    pickle.dump(EEG_X, output_f)

with open('EEG_y.pickle', 'wb') as output_f:
    pickle.dump(EEG_y, output_f)