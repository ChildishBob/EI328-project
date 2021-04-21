import pickle
import numpy as np



with open('EEG_y.pickle', 'rb') as input_f:
    EEG_y = pickle.load(input_f)

EEG_y_onehot = np.empty([EEG_y.shape[0], EEG_y.shape[1], 3], dtype=float)

for i in range(EEG_y.shape[0]):
    for j in range(EEG_y.shape[1]):
        if EEG_y[i][j][0] == -1:
            EEG_y_onehot[i][j] = [1, 0, 0]
        elif EEG_y[i][j][0] == 0:
            EEG_y_onehot[i][j] = [0, 1, 0]
        elif EEG_y[i][j][0] == 1:
            EEG_y_onehot[i][j] = [0, 0, 1]

with open('EEG_y_onehot.pickle', 'wb') as output_f:
    pickle.dump(EEG_y_onehot, output_f)
        