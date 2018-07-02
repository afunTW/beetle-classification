import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from src.callback import LossHistory
from src.core import build_transfer_model
from src.estimate import get_model_memory_usage
from src.loss import focal_loss


def main(args, logger):
    logger.info(args)

    # set up
    logdir = Path('logs')
    if not logdir.exists():
        logdir.mkdir(parents=True)
    log_path = logdir / '{}.log'.format(args.name)
    log_config(log_path, logger)

    outdir = Path('outputs') / args.name
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # build or load model
    resnet = None
    if args.build:
        if args.pretrain_model:
            resnet = build_transfer_model(args.us, args.fs,
                                          n_out=args.n_out,
                                          out_activation=args.out_activation,
                                          model=args.pretrain_model)
            logger.info('Transfer from pretrain model')
        else:    
            resnet = build_transfer_model(args.us, args.fs,
                                          n_out=args.n_out,
                                          out_activation=args.out_activation,
                                          include_top=False,
                                          weights='imagenet',
                                          input_shape=args.input_shape)
            logger.info('Transfer from keras.applications.resnet50 model')
    else:
        assert Path(args.pretrain_model).exists()
        resnet = load_model(args.pretrain_model)
        logger.info('Load model {}'.format(args.pretrain_model))

    optimizer = {'Adam': Adam(lr=args.lr), 'Nadam': Nadam(lr=args.lr)}.get(args.optimizer, None)
    resnet.compile(loss=[focal_loss(gamma=2, alpha=2)], optimizer=optimizer, metrics=['accuracy'])
    estimate_gbytes = get_model_memory_usage(args.bz, resnet)
    plot_model(resnet, to_file=str(outdir / 'model_structure.jpg'))
    logger.info('optimizer = {}'.format(optimizer.__class__.__name__))
    logger.info('Save - {}'.format(str(outdir / 'model_structure.jpg')))
    logger.info('Estimate model required {} gbytes GPU memory'.format(estimate_gbytes))
    logger.info(resnet.summary())

    # preprocess
    h, w, channel = args.input_shape
    count_train_data = len(list(Path(args.train).glob('**/*')))
    count_test_data = len(list(Path(args.test).glob('**/*')))
    logger.info('Got {} training data, {} testing data'.format(count_train_data, count_test_data))
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       featurewise_center=False,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(args.train,
                                                        target_size=(h, w),
                                                        batch_size=args.bz,
                                                        class_mode='categorical',
                                                        shuffle=True)
    test_generator = test_datagen.flow_from_directory(args.test,
                                                      target_size=(h, w),
                                                      batch_size=args.bz,
                                                      class_mode='categorical',
                                                      shuffle=True)
    model_name = outdir / 'resnet_bz{}_us{}_fs{}_{}.h5'.format(
        args.bz, args.us, args.fs, optimizer.__class__.__name__
    )
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-10, verbose=1)
    loss_history = LossHistory()

    # train
    history = resnet.fit_generator(train_generator,
                                   steps_per_epoch=count_train_data // args.bz,
                                   epochs=args.epochs,
                                   validation_data=test_generator,
                                   validation_steps=count_test_data // args.bz,
                                   callbacks=[checkpoint, earlystop, reduce_lr, loss_history])
    history_record = history.history
    history_dataframe = pd.DataFrame(history_record)
    history_dataframe.to_csv(str(outdir / 'history.csv'), index=False)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = argparser()
    main(parser.parse_args(), logger)
