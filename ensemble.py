import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import func_profile, log_handler, plot_confusion_matrix

LOGGER = logging.getLogger(__name__)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method', default='avg', choices=['avg', 'vote'])
    parser.add_argument('--answer', dest='answer', default='data/test', required=True)
    parser.add_argument('--predictions', dest='predictions', nargs='+')
    return parser

def _np_vote(arr1d):
    return np.argmax(np.bincount(arr1d))

def ensemble_avg(y_preds):
    avg_pred = np.average(y_preds, axis=0)
    return np.argmax(avg_pred, axis=-1)

def ensemble_vote(y_preds):
    LOGGER.info('argmax to get voting rather than probability')
    vote_y_pred = np.argmax(y_preds, axis=-1)
    return np.apply_along_axis(_np_vote, 0, vote_y_pred)

def main(args):
    log_handler(LOGGER, logging.getLogger('src.utils'))
    LOGGER.info(args)

    # prepare data
    func_set = {
        'avg': ensemble_avg,
        'vote': ensemble_vote
    }
    X_test = list(Path(args.answer).glob('*/*'))
    y_true = [path.parent.name for path in X_test]
    y_true = np.array(list(map(int, y_true)))
    y_preds = np.array([np.load(i) for i in args.predictions])
    LOGGER.info('predictions shape - {}'.format(y_preds.shape))
    y_pred = func_set.get(args.method)(y_preds)
    LOGGER.info('ensemble predictions shape - {}'.format(y_pred.shape))

    # metrics
    accuracy = accuracy_score(y_pred, y_true, 1)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    savepath_cnf_matrix = str(Path('.') / 'ensemble_confusion_matrix.jpg')
    plot_confusion_matrix(cnf_matrix,
                          classes=list('OX=A'),
                          to_img=savepath_cnf_matrix,
                          normalize=True,
                          title='Confusion matrix (acc={:.4f})'.format(accuracy))
    LOGGER.info('Ensemble top-1-accuracy - {:.4f}'.format(accuracy))
    print(classification_report(y_true, y_pred, target_names=list('OX=A')))
    

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
