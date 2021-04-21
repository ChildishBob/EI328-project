import pickle
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from time import perf_counter

# load data
with open('EEG_X.pickle', 'rb') as input_f:
    EEG_X = pickle.load(input_f)

with open('EEG_y.pickle', 'rb') as input_f:
    EEG_y = pickle.load(input_f)

# print(EEG_X.shape, EEG_X[0].shape, EEG_y.shape, EEG_y[0].shape)

EEG_X_concat = np.concatenate(EEG_X[:])
scaler = StandardScaler()
scaler.fit(EEG_X_concat)

accuracies = []
for i in range(15):
    print(f'======== Round {i} ========')
    test_X = np.reshape(EEG_X[i], (3394, 310))
    test_y = np.reshape(EEG_y[i], 3394)
    train_X = np.reshape(np.concatenate((EEG_X[0:i], EEG_X[i+1:])), (3394*14, 310))
    train_y = np.reshape(np.concatenate((EEG_y[0:i], EEG_y[i+1:])), 3394*14)

    # normalize features
    
    normalized_train_X = scaler.transform(train_X)
    normalized_test_X = scaler.transform(test_X)

    # classify with svm
    clf = LinearSVC(C=0.5)
    clf.fit(normalized_train_X, train_y)
    accuracy = clf.score(normalized_test_X, test_y)
    accuracies.append(accuracy)
    print(f'Accuracy: {accuracy:.4f}')

print('\nAccuracy list:', ''.join(f'{acc:.4f}, ' for acc in accuracies))
print(f'\nMean accuracy: {sum(accuracies)/len(accuracies):.4f}')



