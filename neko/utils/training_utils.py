import numpy as np

from ..layers import Model


def batch_generator(x, y, batch_size=None, shuffle=True, shuffle_seed=None, backend=None):
    """Generates batches from a dataset

    Args:
        x: Input tensor.
        y: Label tensor.
        batch_size: The batch size.
        shuffle: Whether to shuffle the training set for each epoch.
        shuffle_seed: The random seed for shuffling, no seeding if None.
        backend: The backend to use.

    Returns:
        A tuple (x, y)
    """
    n = backend
    samples = x.shape[0]
    if shuffle:
        if shuffle_seed:
            np.random.seed(shuffle_seed)
        perm = np.random.permutation(samples)
    else:
        perm = np.arange(samples)
    full_batches, remainder = divmod(samples, batch_size)
    for i in range(full_batches):
        index_list = perm[batch_size * i: batch_size * (i + 1)]
        yield n.slice_with_list(x, index_list), n.slice_with_list(y, index_list)
    if remainder:
        index_list = perm[batch_size * full_batches:]
        yield n.slice_with_list(x, index_list), n.slice_with_list(y, index_list)


def sync_model_parameters(source, dest):
    assert isinstance(source, Model) and isinstance(dest, Model)
    n = source.backend
    for param_ref, param in zip(source.parameters(), dest.parameters()):
        n.variable_assign(param, n.variable_value(param_ref))
