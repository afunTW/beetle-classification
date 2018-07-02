import numpy as np
from keras import backend as K


def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum(
        [K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum(
        [K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (
        shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0**3), 3)
    return gbytes

def top_n_accuracy(y_pred, y_true, n=1):
    """[summary]
    
    Arguments:
        y_pred {ndarray} -- [description] with shape of n x c which n is instances and c is the number of class
        y_true {ndarray} -- [description] with shape of n x c which n is instances and c is the number of class
    
    Keyword Arguments:
        n {int} -- [description] (default: {1})
    """
    assert n <= y_pred.shape[1]
    y_pred_top_n = np.argsort(y_pred, axis=1)[:, range(y_pred.shape[1]-1, y_pred.shape[1]-1-n, -1)]
    mask = y_pred_top_n == y_true.argmax(axis=1).reshape(len(y_true), 1)
    return sum(mask.any(axis=1))/len(y_pred)
