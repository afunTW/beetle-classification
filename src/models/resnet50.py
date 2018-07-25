import logging
from datetime import datetime

from keras.applications.resnet50 import ResNet50
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from src.models.model import TransferModel

class ResNetTransfer(TransferModel):
    def __init__(self, **config):
        super().__init__(**config)
        self._us = config.get('us', 1)
        self._fs = config.get('fs', 1)
        self._build_transfer_model()

    @property
    def config(self):
        config = super().config
        config.update({
            'us': self._us,
            'fs': self._fs
        })
        return config

    def _extract_layer(self, model):
        assert model is not None
        for layer in model.layers:
            layer.trainable = True
        
        # _layer_map = {'1': 5, '2': 37, '3': 79, '4': 141, '5': 176}
        _layer_name = [layer.name for layer in model.layers]
        _layer_name_map = {
            '1': 'max_pooling2d_1',
            '2': 'activation_10',
            '3': 'activation_22',
            '4': 'activation_40',
            '5': 'avg_pool'
        }
        assert str(self._us) in _layer_name_map

        us_nlayer = _layer_name_map.get(str(self._us))
        us_nlayer = _layer_name.index(us_nlayer)
        fs_nlayer = _layer_name_map.get(str(self._fs), None)
        fs_nlayer = _layer_name.index(fs_nlayer) if fs_nlayer else None

        # freeze
        if fs_nlayer:
            for layer in model.layers[:fs_nlayer+1]:
                layer.trainable = False

        self.logger.info('{} - used {} stage, freeze {} stage'.format(
            self.__class__.__name__, self._us, self._fs
        ))
        x = model.layers[us_nlayer].output
        x = GlobalAveragePooling2D()(x) if 0 < self._us < 5 else x
        return x
    
    def _build_transfer_model(self):
        model = self._pretrain_model
        if not model:
            model = ResNet50(**self._keras_config) if self._keras_config else ResNet50()
        x = self._extract_layer(model)

        # x = Conv2D(512, (1, 1))(x)
        # x = BatchNormalization(axis=-1)(x)

        # x = Conv2D(128, (1, 1))(x)
        # x = BatchNormalization(axis=-1)(x)

        # x = Flatten()(x)

        x = Dense(512)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        x = Dense(256)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Dense(128)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        out_layer = Dense(self._output_nclass, activation=self._output_activation, name='out')(x)
        self.model = Model(inputs=[model.input], outputs=[out_layer])
