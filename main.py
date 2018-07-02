"""[summary]

usage: python3 main.py \
--gpu 0 \
--train data/train \
--test data/test \
--name test \
--comment test trainable model

Returns:
    outputs/README.txt - records the experiment name and detail comments
    outputs/[name] - included log, model, images
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.utils import func_profile, log_handler

LOGGER = logging.getLogger(__name__)
LOGGERS = [LOGGER]

def argparser():
    """[summary]

    --pretrain-model in build mode means the path to save the checkpoint
    --pretrain-model in load mode means the path to load th exists checkpoint
    
    Returns:
        [type] -- [description]
    """

    parser = argparse.ArgumentParser(description='ResNet transfer learning for classification')
    parser.add_argument('--gpu', dest='gpu', required=True)
    parser.add_argument('--train', dest='train', required=True)
    parser.add_argument('--test', dest='test', required=True)
    parser.add_argument('--name', dest='name', default=datetime.now().strftime('%Y%m%d'))
    parser.add_argument('--comment', dest='comment', default='test')
    return parser

def main(args):
    outdir = Path('outputs') / args.name
    if not outdir.exists():
        outdir.mkdir(parents=True)
    log_handler(*LOGGERS, logname=str(outdir / 'log'))

    with open(str(outdir.parent / 'README.txt'), 'a+') as f:
        f.write('{}\t-\t{}\n'.format(args.name, args.comment))


if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
