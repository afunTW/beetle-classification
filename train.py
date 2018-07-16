"""[summary]

usage: python3 train.py \
--gpus 1 \
--train data/train \
--test data/val \
--name test \
--config config/default.json \
--backend resnet \
--ouput-structure \
--comment "test trainable model"

Returns:
    outputs/README.txt - records the experiment name and detail comments
    outputs/[name] - included log, model, images
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, multi_gpu_model
from src.estimate import get_model_memory_usage
from src.loss import focal_loss
from src.models import get_model, get_optimizer
from src.utils import func_profile, log_handler

LOGGER = logging.getLogger(__name__)
LOGGERS = [
    LOGGER,
    logging.getLogger('src.utils')
]

def argparser():
    """[summary]

    --pretrain-model in build mode means the path to save the checkpoint
    --pretrain-model in load mode means the path to load th exists checkpoint
    
    Returns:
        [type] -- [description]
    """

    parser = argparse.ArgumentParser(description='ResNet transfer learning for classification')
    parser.add_argument('--gpus', dest='gpus', required=True, nargs='+')
    parser.add_argument('--train', dest='train', required=True)
    parser.add_argument('--test', dest='test', required=True)
    parser.add_argument('--name', dest='name', default=datetime.now().strftime('%Y%m%d'))
    parser.add_argument('--config', dest='config', default='config/default.json', required=True)
    parser.add_argument('--backend', dest='backend')
    parser.add_argument('--comment', dest='comment', default='test')
    parser.add_argument('--ouput-structure', dest='outimg', action='store_true')
    parser.set_defaults(outimg=False)
    return parser

@func_profile
def main(args):
    # preprocess
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpus)
    outdir = Path('outputs') / args.name
    if not outdir.exists():
        outdir.mkdir(parents=True)
    log_handler(*LOGGERS, logname=str(outdir / 'log.txt'))

    with open(str(outdir.parent / 'README.txt'), 'a+') as f:
        f.write('{}\t-\t{}\n'.format(args.name, args.comment))
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    LOGGER.info(args)
    
    # build and compile model
    K.clear_session()
    model = get_model(args, config)
    model.model = multi_gpu_model(model.model, gpus=len(args.gpus)) if len(args.gpus) > 1 else model.model
    optimizer = get_optimizer(config[args.backend]['optimizer'], lr=config[args.backend]['lr'])
    loss = focal_loss(gamma=2, alpha=2)
    model.model.compile(loss=[loss], optimizer=optimizer, metrics=[config[args.backend]['metrics']])
    estimate_gbytes = get_model_memory_usage(config[args.backend]['bz'], model.model)
    model.model.summary(print_fn=lambda x: LOGGER.info(x + '\n'))
    LOGGER.info('Estimate model required {} gbytes GPU memory'.format(estimate_gbytes))

    # preprocess generator
    _data_gen_params = {
        'target_size': config[args.backend]['keras']['input_shape'][:2],
        'batch_size': config[args.backend]['bz'],
        'class_mode': 'categorical',
        'shuffle': True
    }
    count_train_data = len(list(Path(args.train).glob('**/*')))
    count_test_data = len(list(Path(args.test).glob('**/*')))
    train_data_aug = ImageDataGenerator(**config['imgaug']['train'])
    test_data_aug = ImageDataGenerator(**config['imgaug']['test'])
    train_data_gen = train_data_aug.flow_from_directory(args.train, **_data_gen_params)
    test_data_gen = test_data_aug.flow_from_directory(args.test, **_data_gen_params)
    LOGGER.info('Complete generator preprocess with {} traing data and {} test data'.format(
        count_train_data, count_test_data
    ))

    # callbacks
    model_savepath = str(outdir / '{}.h5'.format(args.backend))
    checkpoint = ModelCheckpoint(model_savepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-10, verbose=1)
    tensorboard = TensorBoard(log_dir=str(outdir / 'tensorboard'),
                            #   histogram_freq=1,
                            #   write_grads=True,
                              write_images=True,
                              write_graph=False)
    LOGGER.info('Complete callbacks declarement')

    # train
    history = model.model.fit_generator(train_data_gen,
                                        steps_per_epoch=count_train_data // config[args.backend]['bz'],
                                        epochs=config[args.backend]['epochs'],
                                        validation_data=test_data_gen,
                                        validation_steps=count_test_data // config[args.backend]['bz'],
                                        callbacks=[checkpoint, earlystop, reducelr, tensorboard])
    
    # save model config
    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.to_csv(str(outdir / 'history.csv'), index=False)
    if args.outimg:
        plot_model(model.model, to_file=str(outdir / 'model_structure.jpg'))
    with open(str(outdir / 'config.json'), 'w+') as f:
        json.dump(model.config, f, indent=4)

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
