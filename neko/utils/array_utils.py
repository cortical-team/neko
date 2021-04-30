import numpy as np

from ..backend import infer_backend_from_tensor


def ensure_list_of_tensor(tensors, backend=None):
    """Convert a list of supported objects to a list of tensors in backend.

    Args:
        tensors: The object to process.
        backend: The backend to use.

    Returns:
        A list of tensors.
    """
    n = backend
    rets = []
    for tensor in tensors:
        if n.is_tensor(tensor):
            rets.append(tensor)
        elif isinstance(tensor, np.ndarray):
            rets.append(n.constant(tensor))
        else:
            raise NotImplementedError(f'tensor of type {type(tensor)} is not supported')
    return rets


def tensors_to_nparray(tensors):
    """Converts a list of tensors, a dictionary of tensors, or just a tensor to numpy array.

    Args:
        tensors: The object to process.

    Returns:
        A numpy array.
    """
    if isinstance(tensors, list):
        return [tensors_to_nparray(t) for t in tensors]
    if isinstance(tensors, dict):
        return {k: tensors_to_nparray(v) for k, v in tensors.items()}
    if isinstance(tensors, np.ndarray) or isinstance(tensors, float) or isinstance(tensors, int):
        return tensors
    n = infer_backend_from_tensor(tensors)
    return n.variable_value_numpy(tensors)
