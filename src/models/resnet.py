import logging
from datetime import datetime

from keras.applications.resnet50 import ResNet50
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam

class ResNet(object):
    def __init__(self, **config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lr = config.get('lr', 1e-2)
        self._bz = config.get('bz', 1)
        self._us = config.get('us', 1)
        self._fs = config.get('fs', 1)
        self._epochs = config.get('epochs', 100)
        self._output_nclass = config.get('output_nclass', 3)
        self._output_activation = config.get('output_activation', 'softmax')
        self._optimizer = config.get('optimizer', 'Adam')
        self._keras_config = config.get('keras')

        self._comment = config.get('comment', '')
        self._build_date = str(datetime.now())
        self._pretrain_model = config.get('pretrain_model')

        self.model = None
        self._build_transfer_model()

    @property
    def config(self):
        return {
            'model': self.__class__.__name__,
            'lr': self._lr,
            'bz': self._bz,
            'us': self._us,
            'fs': self._fs,
            'epochs': self._epochs,
            'output_nclass': self._output_nclass,
            'output_activation': self._output_activation,
            'optimizer': self._optimizer,
            'keras_config': self._keras_config,
            '_comment': self._comment,
            '_build_date': self._build_date
        }

    def _extract_layer(self, model):
        assert model is not None
        for layer in model.layers:
            layer.trainable = True
        
        _layer_map = {'1': 5, '2': 37, '3': 79, '4': 141, '5': 176}
        assert str(self._us) in _layer_map and str(self._fs) in _layer_map

        us_nlayer = _layer_map[str(self._us)]
        fs_nlayer = _layer_map[str(self._fs)]

        # freeze
        for layer in model.layers[:fs_nlayer]:
            layer.trainable = False

        self.logger.info('{} - used {} stage, freeze {} stage'.format(
            self.__class__.__name__, self._us, self._fs
        ))
        x = model.layers[us_nlayer].output
        x = GlobalAveragePooling2D()(x) if 0 < self._us < 5 else x
        return x
    
    def _build_transfer_model(self):
        model = self._pretrain_model
        model = ResNet50(**self._keras_config) if not model and self._keras_config else ResNet50()
        x = self._extract_layer(model)
        out_layer = Dense(self._output_nclass, activation=self._output_activation, name='out')(x)
        self.model = Model(inputs=[model.input], outputs=[out_layer])
