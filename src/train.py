import numpy as np
import os
import pickle
from typing import Tuple

RNG = np.random.RandomState

def set_seed(seed: int = 42):
    np.random.seed(seed)

def one_hot(y, num_classes=None):
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def batch_iter(X, y, batch_size, shuffle=True, rng=None):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        excerpt = indices[start:start + batch_size]
        yield X[excerpt], y[excerpt]

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def save_results(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
