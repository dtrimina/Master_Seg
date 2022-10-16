import torch

#             0,     1,     2,      3,      4,             5,          6,          7,          8,
CLASSES = ('none', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump')


PALETTE = [
            (0, 0, 0),  # unlabelled
            (64, 0, 128),  # car
            (64, 64, 0),  # person
            (0, 128, 192),  # bike
            (0, 0, 192),  # curve
            (128, 128, 0),  # car_stop
            (64, 64, 128),  # guardrail
            (192, 128, 128),  # color_cone
            (192, 64, 0),  # bump
        ]

N_CLASSES = len(CLASSES)



class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


def torch_rand(low, high):
    assert low < high
    return torch.rand(size=(1,)).item() * (high - low) + low


def torch_randint(low, high):
    return torch.randint(low, high, size=(1,)).item()


def torch_rand_choice(options):
    return options[torch_randint(0, len(options))]


def cfg_from_logdir(logdir):
    pass