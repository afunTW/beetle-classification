import logging

LOGGER = logging.getLogger(__name__)

def normalize_generator(batches, backend=None):
    while True:
        batch_x, batch_y = next(batches)

        if backend == 'resnet':
            from keras.applications.resnet50 import preprocess_input
            yield (preprocess_input(batch_x), batch_y)
        elif backend == 'xception':
            from keras.applications.xception import preprocess_input
            yield (preprocess_input(batch_x), batch_y)
        elif backend == 'inception_resnet_v2':
            from keras.applications.inception_resnet_v2 import preprocess_input
            yield (preprocess_input(batch_x), batch_y)
        else:
            yield (batch_x / 255, batch_y)