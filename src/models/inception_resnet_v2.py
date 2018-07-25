import logging
from datetime import datetime

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from src.models.model import TransferModel

class InceptionResNetV2Transfer(TransferModel):
    def __init__(self, **config):
        super().__init__(**config)
        self._build_transfer_model()
    
    def _build_transfer_model(self):
        model = self._pretrain_model
        if not model:
            model = InceptionResNetV2(**self._keras_config) if self._keras_config else InceptionResNetV2()
        
        x = model.layers[-1].output
        x = GlobalAveragePooling2D()(x)

        x = Dense(512)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Dense(128)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        out_layer = Dense(self._output_nclass, activation=self._output_activation, name='out')(x)
        self.model = Model(inputs=[model.input], outputs=[out_layer])