import os
import numpy as np
import pandas as pd
from torch import nn


def get_PAMAP2(path):
    data = np.genfromtxt(path + "\\Protocol\\subject101.dat")
    time_stamps = np.array(data)[:, 0].tolist()  # column 0, time stamps
    y = np.array(data)[:, 1].tolist()  # column 1, labels
    X = np.array(data)[:, 2:]  # the rest of them
    return X, y, time_stamps


def pre_processing(time_window_size, overlap, X):
    if time_window_size < 1:
        print("Error: time_window_size < 1")
        exit(1)

    if time_window_size <= overlap:
        print("Error: time_window_size <= overlap")
        exit(1)

    # replace all nan in X
    X = fill_ndarray(X)

    S = []
    length = X.shape[0]  # how many records

    cursor = 0
    while cursor + time_window_size <= length:
        s_t = X[cursor]
        for i in range(time_window_size - 1):
            s_t = np.concatenate((s_t, X[cursor + i + 1]))
        S.append(s_t)
        cursor = cursor + time_window_size - overlap
    S = np.array(S)

    return S


# fill all NAN with mean value of that col
def fill_ndarray(array):
    for i in range(array.shape[1]):
        temp_col = array[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return array


class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


# Test
PATH = "D:\\Python\\CS589Research\\Data\\PAMAP2_Dataset"
X, y, time_stamps = get_PAMAP2(PATH)
time_window_size = 3
overlap = 0

# X = [[1],
#      [2],
#      [3],
#      [4],
#      [5],
#      [6],
#      [7],
#      [8],
#      [9]
#      ]
# y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# time_stamps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# X = np.array(X)
S = pre_processing(time_window_size, overlap, X)
print(X.shape)
print(S.shape)


