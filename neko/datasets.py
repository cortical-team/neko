import os
from abc import abstractmethod

import numpy as np

from .backend import numpy_backend as npb
from .utils.network_utils import download_file

_dataset_dir = os.path.join(os.path.expanduser('~'), '.neko', 'datasets')


class Dataset:
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        raise NotImplementedError


class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
        self.filename = 'mnist.npz'

    def load(self, onehot=True):
        """Loads MNIST dataset.

        Args:
            onehot: Whether to use onehot vector as targets.

        Returns:
            A tuple (x_train, y_train, x_test, y_test)
        """
        download_file(self.url, _dataset_dir, self.filename)
        dest_file = os.path.join(_dataset_dir, self.filename)
        with np.load(dest_file) as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
            x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
            if onehot:
                y_train = npb.categorical_to_onehot(y_train, 10).astype(np.float32)
                y_test = npb.categorical_to_onehot(y_test, 10).astype(np.float32)
            else:
                y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
            return x_train, y_train, x_test, y_test
