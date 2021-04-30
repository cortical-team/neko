_activation_registry = {'elu', 'exponential', 'hard_sigmoid', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid',
                        'softmax', 'softplus', 'softsign', 'tanh', 'heaviside', 'heaviside2'}


def get_activation(name, backend):
    """Converts a string specification of activation to the corresponding function.

    Args:
        name: The string specification.
        backend: The backend to use.

    Returns:
        The activation function.
    """
    n = backend
    name = name.lower()
    if name in _activation_registry:
        return getattr(n, name)
    else:
        raise Exception(f'Activation function {name} is not supported.')
