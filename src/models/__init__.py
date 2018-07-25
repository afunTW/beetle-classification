from keras.optimizers import Adam, Nadam, SGD


def get_model(args, config):
    if args.backend in config:
        if args.backend == 'resnet':
            from .resnet50 import ResNetTransfer
            return ResNetTransfer(comment=args.comment, **config[args.backend])
        elif args.backend == 'xception':
            from .xception import XceptionTransfer
            return XceptionTransfer(comment=args.comment, **config[args.backend])
        elif args.backend == 'inception_resnet_v2':
            from .inception_resnet_v2 import InceptionResNetV2Transfer
            return InceptionResNetV2Transfer(comment=args.comment, **config[args.backend])

def get_optimizer(name, **params):
    _optimizer_map = {
        'Adam': Adam(**params),
        'Nadam': Nadam(**params),
        'SGD': SGD(**params)
    }
    return _optimizer_map.get(name)
