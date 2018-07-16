import logging
from datetime import datetime

from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model
from src.models.model import TransferModel


class XceptionTransfer(TransferModel):
    def __init__(self, **config):
        super().__init__(**config)
        self._build_transfer_model()
    
    def _build_transfer_model(self):
        model = self._pretrain_model
        if not model:
            model = Xception(**self._keras_config) if self._keras_config else Xception()
        
        x = model.layers[-1].output
        x = GlobalAveragePooling2D()(x)

        out_layer = Dense(self._output_nclass, activation=self._output_activation, name='out')(x)
        self.model = Model(inputs=[model.input], outputs=[out_layer])
