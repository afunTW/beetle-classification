import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

def top_n_accuracy(y_pred, y_true, n):
    """
    y_pred: prediction, np.array with shape of n x c which n is instances and c is the number of class
    y_true: groudtruth label, np.array with shape of n x c which n is instances and c is the number of class
    n: top n
    """
    assert n <= y_pred.shape[1]
    y_pred_top_n = np.argsort(y_pred, axis=1)[:, range(y_pred.shape[1]-1, y_pred.shape[1]-1-n, -1)]
    mask = y_pred_top_n == y_true.argmax(axis=1).reshape(len(y_true), 1)
    return sum(mask.any(axis=1))/len(y_pred)
