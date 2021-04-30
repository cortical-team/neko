from functools import partial
from math import prod

from .losses import _loss_registry, get_loss


def accuracy(*, model, x, y_true, from_logits=True, backend=None, **kwargs):
    """Accuracy for classification tasks.

    Args:
        model: The layer instance.
        x: Input tensor
        y_true: Label tensor, placeholder only, not used.
        from_logits: True if the model does not include a softmax in its forward calculation.
        backend: The backend to use.
        **kwargs: Placeholder to accept and discard parameters for other metric functions.

    Returns:
        A numpy array.
    """
    n = backend
    y_pred = model(x)

    assert len(y_pred.shape) >= 2, f'predictions has wrong shape {len(y_pred.shape)}'

    if from_logits:
        y_pred = n.softmax(y_pred)

    y_pred_class = n.argmax(y_pred, -1)
    # when y_true has dim>=3, assume it is categorical
    if len(y_true.shape) >= 2:
        # categorical accuracy
        # y_pred = [[0.4, 0.6], [0.2, 0.8]]; y_true = [[0., 1.], [1., 0.]]
        y_true_class = n.argmax(y_true, -1)
    elif len(y_true.shape) == 1:
        # sparse categorical accuracy
        # y_pred = [[0.4, 0.6], [0.2, 0.8]]; y_true = [0, 1]
        y_true_class = y_true
    else:
        assert False, f'label has unsupported shape {y_true.shape}'
    corrects = n.sum(n.cast(y_pred_class == y_true_class, n.int32))
    return n.variable_value_numpy(corrects) / prod(y_true.shape[:-1])


def firing_rate(*, model, x, y_true, backend=None, **kwargs):
    """Displays the firing rate for a model if applicable.

    Args:
        model: The layer instance.
        x: Input tensor
        y_true: Label tensor, placeholder only, not used.
        backend: The backend to use.
        **kwargs: Placeholder to accept and discard parameters for other metric functions.

    Returns:
        A tensor.
    """
    n = backend
    output_dict = model.forward(x, return_internals=True)
    z = output_dict['z']
    _, n_timestep, _ = z.shape
    return n.reduce_mean(n.einsum('btj->bj', z) / n_timestep / model.simulation_interval)


_metric_registry_dict = {'accuracy': accuracy, 'firing_rate': firing_rate}


def get_metric(name, backend, **kwargs):
    """Converts a string specification of metric function to the corresponding function.

    Args:
        name: The string specification.
        backend: The backend to use.
        **kwargs: Keyword arguments to pass to metric functions.

    Returns:
        The metric function.
    """
    n = backend
    name = name.lower()
    if name in _loss_registry:
        return get_loss(name, backend, **kwargs)
    elif name in _metric_registry_dict:
        return partial(_metric_registry_dict[name], backend=n, **kwargs)
    else:
        raise Exception(f'metric {name} is not supported.')
