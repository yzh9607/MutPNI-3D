import math
from functools import partial
from typing import List, Union, Any

import numpy as np
import torch
import torch.nn as nn
import functools

from torch.nn import ReLU, LeakyReLU, PReLU, Tanh, SELU, ELU


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_func(activation: str):
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation is None:
        return None
    elif activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.2)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "swish":
        return functools.partial(swish)
    else:
        raise ValueError(f'Activation "{activation}" not supported.')







