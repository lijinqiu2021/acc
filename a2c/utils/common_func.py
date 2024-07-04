import collections
import random

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations for CuDNN, it may impact performances (from stable baseline3)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def synthesize(array):
    d = collections.OrderedDict()
    if len(array) == 0:
        d["mean"] = 0
        d["std"] = 0
        d["min"] = 0
        d["max"] = 0
    else:
        d["mean"] = np.mean(array)
        d["std"] = np.std(array)
        d["min"] = np.min(array)
        d["max"] = np.max(array)
    return d
