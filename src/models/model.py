import logging
from datetime import datetime


class TransferModel(object):
    def __init__(self, **config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

        self._lr = config.get('lr', 1e-2)
        self._bz = config.get('bz', 1)
        self._epochs = config.get('epochs', 100)
        self._output_nclass = config.get('output_nclass', 3)
        self._output_activation = config.get('output_activation', 'softmax')
        self._optimizer = config.get('optimizer', 'Adam')
        self._keras_config = config.get('keras')

        self._comment = config.get('comment', '')
        self._build_date = str(datetime.now())
        self._pretrain_model = config.get('pretrain_model')
    
    @property
    def config(self):
        return {
            'model': self.__class__.__name__,
            'lr': self._lr,
            'bz': self._bz,
            'epochs': self._epochs,
            'output_nclass': self._output_nclass,
            'output_activation': self._output_activation,
            'optimizer': self._optimizer,
            'keras_config': self._keras_config,
            '_comment': self._comment,
            '_build_date': self._build_date
        }
