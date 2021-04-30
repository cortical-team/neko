from abc import abstractmethod, ABC

from ..initializers import get_initializer


class Model(ABC):
    def __init__(self, backend=None):
        """

        Args:
            backend: backend object
        """
        self.parameters_dict = {}
        self.initializer = None
        self.initialized = False
        self.seed = None
        self.backend = backend

        assert self.backend, 'no backend has been set'

    def parameters(self):
        """

        Returns:
            A list of layer parameters
        """
        return list(self.parameters_dict.values())

    def add_parameter(self, name, shape):
        """Add and register a parameter to the layer.

        Args:
            name: A string which is the name of the parameter
            shape: A tuple indicating the shape of the parameter

        Returns:
            None
        """
        n = self.backend
        setattr(self, name, n.variable(shape=shape, name=name))
        self.parameters_dict[name] = getattr(self, name)

    def set_parameter(self, name, data):
        """Assign a value to a parameter.

        Args:
            name: The name of the parameter, must be already initialized
            data: The value to be assigned

        Returns:
            None
        """
        n = self.backend
        parameter = getattr(self, name)
        assert list(data.shape) == list(parameter.shape)
        n.variable_assign(tensor=parameter, data=data)

    def initialize_parameters(self):
        """Initialize the parameters in the layer according to the initializer string, function, or value.

        Returns:
            None
        """
        if not self.initializer:
            # default uniform initializer
            self.initializer = {'default': get_initializer('glorot_uniform')(seed=self.seed),
                                'w_hh': get_initializer('orthogonal')(seed=self.seed),
                                'b_ho': get_initializer('zeros')()}

        # case 1, initializer is a function, apply to all parameters
        if callable(self.initializer):
            if isinstance(self.initializer, type):
                self.initializer = self.initializer(seed=self.seed)
            for name, param in self.parameters_dict.items():
                self.set_parameter(name, self.initializer(shape=param.shape))

        # case 2, initialization dictionary, can be value or function
        if isinstance(self.initializer, dict):
            default_initializer = False
            if 'default' in self.initializer:
                default_initializer = self.initializer['default']

            for name, param in self.parameters_dict.items():
                if name in self.initializer:
                    if callable(self.initializer[name]):
                        if isinstance(self.initializer[name], type):
                            self.initializer[name] = self.initializer[name](seed=self.seed)
                        self.set_parameter(name, self.initializer[name](shape=param.shape))
                    else:
                        self.set_parameter(name, self.initializer[name])
                elif default_initializer:
                    if callable(default_initializer):
                        if isinstance(default_initializer, type):
                            default_initializer = default_initializer(seed=self.seed)
                        self.set_parameter(name, default_initializer(shape=param.shape))
                    else:
                        self.set_parameter(name, default_initializer)
                else:
                    pass

    def check_or_build(self, inputs):
        """Check if a layer has been initialized. If not, perform the initialization.

        Args:
            inputs: Input tensor of a network.

        Returns:
            None
        """
        if not self.backend:
            raise Exception('No backend specified!')
        if not self.initialized:
            self.build(inputs.shape)
            self.initialize_parameters()
            self.initialized = True

    def __call__(self, inputs):
        """Default function call of the layer

        Args:
            inputs: Input tensor

        Returns:
            Network output tensor
        """
        self.check_or_build(inputs)
        return self.forward(inputs)

    @abstractmethod
    def build(self, input_shape):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    def fit(self, x, y, learning_rule, **kwargs):
        self.check_or_build(x)
        return learning_rule.train(self, x, y, backend=self.backend, **kwargs)

    def predict(self, inputs):
        """Same as a forward call.

        Args:
            inputs: Input tensor

        Returns:
            Network output tensor
        """
        return self.forward(inputs)

    def predict_class(self, inputs):
        """Output prediction in classes instead of probabilities.

        Args:
            inputs: Input tensor

        Returns:
            Network output tensor
        """
        n = self.backend
        return n.argmax(self.forward(inputs), 1)


class Epropable(ABC):
    """An interface that must be implemented by layers compatible with eprop algorithms.

    """

    def __init__(self, backend=None):
        super().__init__(backend)

    @abstractmethod
    def dht__dht_1(self):
        """Calculates the Jacobian dh[t]/dh[t-1]

        Returns:
            A function that returns a n*n matrix, where n is the hidden state size of a neuron.
            The returned function returns a tensor of shape (batch_size, hidden_state_dims, hidden_state_dims, n_hidden_neuron).
        """
        pass

    @abstractmethod
    def dht__dWhh(self):
        """Calculates the dh[t]/dW_{hh}

        Returns:
            A function that returns a vector of length n, where n is the hidden state size of a neuron.
            The returned function returns a tensor of shape (batch_size, hidden_state_dims, n_hidden_neuron).
        """
        pass

    @abstractmethod
    def dht__dWih(self):
        """Calculates the dh[t]/dW_{ih}

        Returns:
            A function that returns a vector of length n, where n is the hidden state size of a neuron.
            The returned function returns a tensor of shape (batch_size, hidden_state_dims, input_size).
        """
        pass

    @abstractmethod
    def dzt__dht(self):
        """Calculates the dz[t]/dh[t]

        Returns:
            A function that returns a tuple. The first element is usually a pseudo-derivative of shape (batch_size, n_hidden_neuron),
            and the second element is a vector of length hidden_state_dims, which are the coefficients of the derivative.
        """
        pass
