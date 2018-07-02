import logging

from keras.applications.resnet50 import ResNet50
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

logger = logging.getLogger(__name__)

def extract_layer(model, us, fs):
    for layer in model.layers:
        layer.trainable = True

    layer_map = {
        '1': 5,
        '2': 37,
        '3': 79,
        '4': 141,
        '5': 176
    }

    assert str(us) in layer_map.keys()
    assert str(fs) in layer_map.keys()

    nth_us_layer = layer_map[str(us)]
    nth_fz_layer = layer_map[str(fs)]

    # freeze layer
    for layer in model.layers[:nth_fz_layer]:
        layer.trainable = False

    logger.info('used stage: {}, freeze stage: {}'.format(us, fs))
    x = model.layers[nth_us_layer].output
    x = GlobalAveragePooling2D()(x) if 0 < us < 5 else x
    return x

def build_transfer_model(us, fs, n_out,
                         out_activation='linear',
                         n_meta=0, n_meta_dense=1024,
                         model=None,
                         **params):
    """[summary] Transfer ResNet Model
    - define resnet
    - define transfer
    - define output structure

    Arguments:
        us {int} -- used stage for resnet transfer learning
        fs {int} -- freeze stage for resnet transfer learning
        n_out {int} -- number of output neuron

    Keyword Arguments:
        out_activation {str} -- output activation func (default: {'linear'})
        n_meta {int} -- #metadata (default: {0})
        n_meta_dense {int} -- #neuron on Dense layer after concat metadata  (default: {1024})
        model {[type]} -- should be keras model (default: {None})
        params {[type]} -- keras ResNet model parameters
    """
    model = model or ResNet50(**params)
    x = extract_layer(model, us, fs)

    # metadata
    in_meta = None
    if n_meta > 0:
        in_meta = Input(shape=(n_meta,))
        x = concatenate([x, in_meta])
        x = Dense(n_meta_dense)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

    # output
    out_layer = Dense(n_out, activation=out_activation, name='out')(x)
    if n_meta > 0 and not in_meta:
        return Model(inputs=[model.input, in_meta], outputs=[out_layer])
    else:
        return Model(inputs=[model.input], outputs=[out_layer])
