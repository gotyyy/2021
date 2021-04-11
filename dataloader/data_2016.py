import os,random, pickle
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.utils import np_utils
import math

def data_load(filepath):
    # 读取数据
    Xd = pickle.load(open(filepath, 'rb'), encoding='latin1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    print(X.shape)

    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(round(n_examples * 0.5))
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))

    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    # transform iq to ap
    X_train_AP = AP(X_train)
    X_test_AP = AP(X_test)


    in_shp = list(X_train.shape[1:])
    print(X_train.shape, in_shp)
    classes = mods

    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))

    return X_train_AP,Y_train,X_test_AP,Y_test,classes, snrs,test_SNRs


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1  # ?
    return yy1

# Define the transformation(A/P)
def AP(x):
    X_aps = []
    A = x.shape[0]
    for a in range(0, A):
        x_ap = []
        for b in range(128):
            x_a = math.atan2(x[a, 0, b], x[a, 1, b])
            x_p = x[a, 0, b]
            x_1 = (x_a, x_p)
            x_ap.append(x_1)
        X_aps.append(x_ap)

    return np.array(X_aps).transpose((0, 2, 1))