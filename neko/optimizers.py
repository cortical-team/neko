from .backend import infer_backend_from_tensor


class Optimizer:
    def __init__(self):
        self.backend = None
        self.optimizer = None
        self.parameters = None

    def initialized(self):
        """Internal routine to check if the optimizer is initialized. Fails if watch() is not called.

        Returns:
            None
        """
        return self.backend and self.optimizer

    def _infer_and_set_backend(self, parameter):
        """

        Args:
            parameter: Parameters to optimize.

        Returns:
            None
        """
        self.backend = infer_backend_from_tensor(parameter)

    def watch(self, parameters):
        """Specify the parameters to optimize. It must be called before optimization.

        Args:
            parameters: A list of parameters to optimize.

        Returns:

        """
        assert isinstance(parameters, list), 'parameters should be a list'
        assert len(parameters) > 0, 'Empty parameter list for optimizer'
        self.parameters = parameters
        self._infer_and_set_backend(parameters[0])

    def calculate_gradients_and_optimize(self, network_objective):
        """Calculates gradients and apply them in the classic backprop algorithm.

        Args:
            network_objective: A callable that returns the network objective, usually the loss.

        Returns:
            The loss
        """
        assert self.initialized(), 'Optimizer not initialized'
        assert callable(network_objective)
        n = self.backend
        return n.calculate_gradients_and_optimize(self.parameters, self.optimizer, network_objective)

    def loss_and_gradients(self, network_objective):
        """Calculates gradients but do not apply them.

        Args:
            network_objective: A callable that returns the network objective, usually the loss.

        Returns:
            A list of tensor having the same shapes of self.parameters
        """
        assert self.backend
        assert callable(network_objective)
        n = self.backend
        return n.loss_and_gradients(self.parameters, network_objective)

    def apply_gradients(self, parameters, gradients):
        """Applies custom gradients to parameters with the optimizer, no backprop is performed.

        Args:
            parameters: A list of parameters to optimize.
            gradients: Corresponding list of gradient changes.

        Returns:
            None
        """
        assert self.initialized(), 'Optimizer not initialized'
        n = self.backend
        return n.apply_custom_gradients(parameters, gradients, self.optimizer)

    def change_parameter(self, parameter_name, parameter_value):
        """Change a parameter of the optimizer, like the learning rate. Not to be confused with network parameters.

        Args:
            parameter_name: string, the name of the parameter to change
            parameter_value: float, the value

        Returns:
            None
        """
        assert self.initialized(), 'Optimizer not initialized'
        n = self.backend
        n.optimizer_change_parameter(self.optimizer, parameter_name, parameter_value)


class NaiveOptimizer(Optimizer):
    def __init__(self):
        super().__init__()

    def apply_gradients(self, parameters, gradients):
        assert self.backend
        n = self.backend
        for p, g in zip(parameters, gradients):
            n.variable_assign(p, p + g)


class Adadelta(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-07):
        # pytorch params preset: 1, 0.9, 1e-6
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.adadelta(parameters, self.learning_rate, self.rho, self.epsilon)


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07):
        # pytorch params preset: 0.01, 0, 1e-10
        super().__init__()
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.adagrad(parameters, self.learning_rate, self.initial_accumulator_value, self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.adam(parameters, self.learning_rate, self.beta_1, self.beta_2, self.epsilon, self.amsgrad)


class Adamax(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.adamax(parameters, self.learning_rate, self.beta_1, self.beta_2, self.epsilon)


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False):
        # pytorch, 0.01, 0.99, 0, 1e-8
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.rmsprop(parameters, self.learning_rate, self.rho, self.momentum, self.epsilon, self.centered)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def watch(self, parameters):
        super().watch(parameters)
        n = self.backend
        self.optimizer = n.sgd(parameters, self.learning_rate, self.momentum, self.nesterov)


_optimizer_registry = {'adadelta': Adadelta, 'adagrad': Adagrad, 'adam': Adam, 'adamax': Adamax, 'rmsprop': RMSprop,
                       'sgd': SGD}


def get_optimizer(name):
    """Converts a string specification of an optimizer to the corresponding class.

    Args:
        name: The optimizer name

    Returns:
        An optimizer instance
    """
    name = name.lower()
    if name in _optimizer_registry:
        return _optimizer_registry[name]()
    else:
        raise Exception(f'Optimizer {name} is not supported.')
