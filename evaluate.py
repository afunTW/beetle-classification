import argparse
import logging
import os
from pathlib import Path

from PIL import Image

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from src.loss import focal_loss
from src.metrics import top_n_accuracy
from src.utils import func_profile, log_handler, plot_confusion_matrix


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', dest='gpus', required=True)
    parser.add_argument('--models', dest='models', required=True, nargs='+')
    parser.add_argument('--test', dest='test', default='data/test')
    parser.add_argument('--focal-loss', dest='focal_loss', action='store_true')
    parser.add_argument('--no-focal-loss', dest='focal_loss', action='store_false')
    parser.set_defaults(focal_loss=True)
    return parser

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

    # models
    for model_path in args.models:
        K.clear_session()
        if args.focal_loss:
            model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss(2, 2)})
        else:
            model = load_model(model_path)

        # predict
        y_pred = model.predict(X_test)

        # metrics
        accuracy = top_n_accuracy(y_pred, y_true, 1)
        cnf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        savepath_cnf_matrix = str(Path(model_path).with_name('confusion_matrix.jpg'))
        plot_confusion_matrix(cnf_matrix,
                              classes=list('OX=A'),
                              to_img=savepath_cnf_matrix,
                              normalize=True,
                              title='Confusion matrix (acc={:2f})'.format(accuracy))
        logger.info('model: {}'.format(model_path))
        logger.info(' - top-1-accuracy: {:.4f}'.format(accuracy))
        logger.info(' - save confusion matrix at {}'.format(savepath_cnf_matrix))
        print(classification_report(y_true.argmax(axis=1),
                                    y_pred.argmax(axis=1),
                                    target_names=list('OX=A')))

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
