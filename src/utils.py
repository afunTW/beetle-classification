import itertools
import logging
import sys
from datetime import datetime
from functools import wraps

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

LOGGER = logging.getLogger(__name__)


def log_handler(*loggers, logname=None):
    formatter = logging.Formatter(
        '%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # file handler
    if logname:
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

    for logger in loggers:
        if logname:
            logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)

def func_profile(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        cost_time = datetime.now() - start_time
        fullname = '{}.{}'.format(func.__module__, func.__name__)
        LOGGER.info('{}[kwargs={}] completed in {}'.format(
            fullname, kwargs, str(cost_time)
        ))
        return result
    return wrapped

def plot_confusion_matrix(cm, classes, to_img,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        LOGGER.info('Normalized confusion matrix')
    else:
        LOGGER.info('Confusion matrix, without normalization')

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(to_img)
