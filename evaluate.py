import os
import argparse
import logging
import numpy as np

from PIL import Image
from pathlib import Path
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from src.utils import func_profile, log_handler
from src.loss import focal_loss

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', dest='gpus', required=True)
    parser.add_argument('--model', dest='model', required=True)
    parser.add_argument('--test', dest='test', default='data/test')
    parser.add_argument('--focal-loss', dest='focal_loss', action='store_true')
    parser.add_argument('--no-focal-loss', dest='focal_loss', action='store_false')
    parser.set_defaults(focal_loss=True)
    return parser

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

@func_profile
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logger = logging.getLogger(__name__)
    log_handler(logger, logging.getLogger('src.utils'))
    logger.info(args)

    # prepare data
    X_test = list(Path(args.test).glob('*/*'))
    y_true = [path.parent.name for path in X_test]
    y_true = list(map(int, y_true))
    y_true = np_utils.to_categorical(y_true)
    X_test = np.array([np.array(Image.open(i).resize((224, 224)), dtype='float32') for i in X_test])
    X_test = X_test / 255
    logger.info('X_test shape={}, Y_test shape={}'.format(X_test.shape, y_true.shape))

    # model
    K.clear_session()
    if args.focal_loss:
        model = load_model(args.model, custom_objects={'focal_loss_fixed': focal_loss(2, 2)})
    else:
        model = load_model(args.model)

    # predict
    y_predict = model.predict(X_test)

    # metrics
    accuracy = top_n_accuracy(y_predict, y_true, 1)
    logger.info('top-1-accuracy: {:.4f}'.format(accuracy))

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
