"""
This file is a slightly modified version of the file, of the same name, found on the openai repo:
https://github.com/openai/generating-reviews-discovering-sentiment
"""

import html
import os

import numpy as np


def load_model_params(params_path):
    """Load the pretrained weights (numpy arrays) from disk."""
    num_npy_arrays = 15
    params = [np.load(os.path.join(params_path, '%d.npy' % i)) for i in range(num_npy_arrays)]
    params[2] = np.concatenate(params[2:6], axis=1)
    params[3:6] = []
    return params


def preprocess(text, front_pad='\n ', end_pad=' '):
    """Preprocess the sentences.

    Newlines need to be stripped to avoid resetting the model state. Also, a start and end token are
    simulated with a newline + space and a space, respectively.
    """
    text = html.unescape(text)
    text = text.replace('\n', ' ').strip()
    text = front_pad + text + end_pad
    text = text.encode()
    return text


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)

    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]

    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HyperParams(object):
    """Collects and sets model hyper parameters."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
