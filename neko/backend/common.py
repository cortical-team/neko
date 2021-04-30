from ..backend import numpy_backend as npb
from ..backend import pytorch_backend as pytb
from ..backend import tensorflow_backend as tfb


def infer_backend_from_tensor(tensor):
    if npb.is_tensor(tensor):
        return npb
    elif tfb.is_tensor(tensor):
        return tfb
    elif pytb.is_tensor(tensor):
        return pytb
