from keras.optimizers import Adam, Nadam, SGD


def get_model(args, config):
    if args.backend in config:
        if args.backend == 'resnet':
            from .resnet import ResNet
            return ResNet(comment=args.comment, **config[args.backend])

def get_optimizer(name, **params):
    _optimizer_map = {
        'Adam': Adam(**params),
        'Nadam': Nadam(**params),
        'SGD': SGD(**params)
    }
    return _optimizer_map.get(name)