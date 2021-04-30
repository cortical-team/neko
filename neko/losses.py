from functools import partial

_loss_registry = {'sparse_categorical_crossentropy', 'categorical_crossentropy', 'kld', 'kullback_leibler_divergence',
                  'mae', 'mean_absolute_error', 'mean_squared_error', 'mse', 'poisson'}


def firing_rate_regularization(*, model, x, y_true, firing_rate_target=10, backend=None, **kwargs):
    """MSE loss of firing rate used in regularization.

    Args:
        model: The layer instance.
        x: Input tensor
        y_true: Label tensor, placeholder only, not used.
        firing_rate_target: Target of firing rate in Hz.
        backend: The backend to use.
        **kwargs: Placeholder to accept and discard parameters for other loss functions.

    Returns:
        A tensor.
    """
    n = backend
    output_dict = model.forward(x, return_internals=True)
    z = output_dict['z']
    _, n_timestep, _ = z.shape
    return n.reduce_mean(
        0.5 * n.square(n.einsum('btj->bj', z) / n_timestep / model.simulation_interval - firing_rate_target))


_loss_registry_dict = {'firing_rate_regularization': firing_rate_regularization}


def _loss_adaptor(*, model, x, y_true, loss_fn, **kwargs):
    y_pred = model(x)
    return loss_fn(y_true=y_true, y_pred=y_pred)


def get_loss(name, backend, **kwargs):
    """Converts a string specification of loss function to the corresponding function.

    Args:
        name: The string specification.
        backend: The backend to use.
        **kwargs: Keyword arguments to pass to loss functions.

    Returns:
        The loss function.
    """
    n = backend
    name = name.lower()
    if name in _loss_registry_dict:
        return partial(_loss_registry_dict[name], backend=n, **kwargs)
    elif name in _loss_registry:
        return partial(_loss_adaptor, loss_fn=getattr(n, name))
    else:
        raise Exception(f'Loss function {name} is not supported.')
