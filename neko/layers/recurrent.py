import math
from abc import ABC, abstractmethod

from .base import Model, Epropable
from ..activations import get_activation


class RecurrentBaseModel(Model, ABC):
    def __init__(self,
                 hidden_size,
                 output_size=1,
                 initializer=None,
                 activation=None,
                 use_readout_bias=None,
                 return_sequence=True,
                 task_type=None,
                 seed=None,
                 backend=None):
        """

        Args:
            hidden_size: Number of hidden neurons
            output_size: Dimension of network output (target)
            initializer: A global initializer specification or a dictionary of initializer specifications
                with parameters as key (key 'default' for unspecified parameters). An initializer specification
                can be a string or a function.
            activation: Activation function in the recurrent layer. It can be a string or function.
            use_readout_bias: Whether to use a bias term on output.
            return_sequence: Whether to return outputs from all timesteps.
            task_type: 'regression' or 'classification'. Their losses will be MSE and softmax with cross entropy respectively
            seed: The random seed, no seeding if None.
            backend: The backend object to use.
        """
        super().__init__(backend)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.d_activation = None
        self.use_readout_bias = use_readout_bias
        self.return_sequence = return_sequence
        self.task_type = task_type
        self.initializer = initializer
        self.seed = seed
        self.input_size = None

        if isinstance(self.activation, str):
            self.activation = get_activation(self.activation, backend)
        if not self.task_type:
            self.task_type = 'regression'

    def build(self, input_shape):
        """Registers the parameter of the layer according to the input shape.

        Args:
            input_shape: A tuple or class that can be unpacked to three elements, indicating the dimensions of the input.

        Returns:
            None
        """
        _, _, self.input_size = input_shape
        # synaptic weight matrices
        self.add_parameter('w_ih', shape=(self.input_size, self.hidden_size))
        self.add_parameter('w_hh', shape=(self.hidden_size, self.hidden_size))
        self.add_parameter('w_ho', shape=(self.hidden_size, self.output_size))
        if self.use_readout_bias:
            self.add_parameter('b_ho', shape=(self.output_size,))

    def _process_inputs(self, inputs, n_timestep):
        """Pad or clip the length of input to be consistent with the simulation steps.

        Args:
            inputs: Input tensor
            n_timestep: Number of timestep/simulation steps.

        Returns:
            Processed tensor.
        """
        n = self.backend
        # input vector's shape should be [batch_size, n_inputs, input_size]
        batch_size, current_n_timestep, input_size = inputs.shape

        # if n_step is None, just return
        if not n_timestep:
            return inputs

        # if input length is longer than n_step, we should truncate it, otherwise we should pad it with zeros
        if current_n_timestep >= n_timestep:
            x = inputs[:, n_timestep, :]
        else:
            x = n.zeros([batch_size, n_timestep, input_size])
            x[:, current_n_timestep, :] = inputs
        return x

    def _get_d_activation(self):
        """Helper function to get the derivative of the activation function, for learning algorithms which do not
            use auto differentiation.

        Returns:
            A function which is the derivative of the activation.
        """
        n = self.backend
        if self.d_activation:
            return self.d_activation
        else:
            return n.get_derivative(self.activation)

    @abstractmethod
    def forward(self, inputs, n_timestep=None, *, return_internals=False):
        """Forward interface for recurrent layers.

        Args:
            inputs: Input tensor
            n_timestep: Number of timesteps
            return_internals: Whether to return the result, or a dictionary of all internal states.

        Returns:
            A tensor or dictionary.
            The dictionary will contain 'h' of shape (batch_size, n_timestep, hidden_state_dims, n_hidden_neuron),
            'z' of shape (batch_size, n_timestep, n_hidden_neuron),
            'output_sequence' of shape (batch_size, n_timestep, output_size), and the return value.
        """
        raise NotImplementedError()


class BasicRNNModel(RecurrentBaseModel, Epropable):
    def __init__(self,
                 hidden_size,
                 output_size=1,
                 initializer=None,
                 activation='tanh',
                 simulation_interval=math.inf,
                 recurrent_membrane_time_constant=0.02,
                 output_membrane_time_constant=0.02,
                 alpha=None,
                 kappa=None,
                 use_bias=False,
                 use_readout_bias=True,
                 return_sequence=True,
                 task_type='regression',
                 seed=None,
                 backend=None,
                 **kwargs):
        """

        Args:
            hidden_size: Number of hidden neurons
            output_size: Dimension of network output (target)
            initializer: A global initializer specification or a dictionary of initializer specifications
                with parameters as key (key 'default' for unspecified parameters). An initializer specification
                can be a string or a function.
            activation: Activation function in the recurrent layer. It can be a string or function.
            simulation_interval: An artificial parameter for regular RNN, defaults to infinity to disable decays.
            recurrent_membrane_time_constant: Membrane time constant for recurrent (hidden) neurons.
            output_membrane_time_constant: Membrane time constant for output neurons.
            alpha: The decay factor of recurrent units, will override calculations from membrane time constant if specified.
            kappa: The decay factor of output units, will override calculations from membrane time constant if specified.
            use_bias: Whether to use a bias term on recurrent units.
            use_readout_bias: Whether to use a bias term on output.
            return_sequence: Whether to return outputs from all timesteps.
            task_type: 'regression' or 'classification'. Their losses will be MSE and softmax with cross entropy respectively
            seed: The random seed, no seeding if None.
            backend: The backend object to use.
            **kwargs: Placeholder to accept and discard parameters for other types of layers.
        """
        super().__init__(hidden_size=hidden_size,
                         output_size=output_size,
                         initializer=initializer,
                         activation=activation,
                         use_readout_bias=use_readout_bias,
                         return_sequence=return_sequence,
                         task_type=task_type,
                         seed=seed,
                         backend=backend)
        self.use_bias = use_bias
        n = self.backend
        if not alpha:
            alpha = math.exp(- simulation_interval / recurrent_membrane_time_constant)
        if not kappa:
            kappa = math.exp(- simulation_interval / output_membrane_time_constant)
        self.alpha = n.constant(alpha, dtype=n.float32)
        self.kappa = n.constant(kappa, dtype=n.float32)
        self.simulation_interval = n.constant(simulation_interval, dtype=n.float32)
        self.recurrent_membrane_time_constant = n.constant(recurrent_membrane_time_constant, dtype=n.float32)
        self.output_membrane_time_constant = n.constant(output_membrane_time_constant, dtype=n.float32)
        self.initial_state = None

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_bias:
            self.add_parameter('b_hh', shape=(self.hidden_size,))

    def forward(self, inputs, n_timestep=None, *, return_internals=False):
        n = self.backend
        x = self._process_inputs(inputs, n_timestep)
        batch_size, n_timestep, _ = x.shape

        hidden_states = []
        fake_spikes = []
        outputs = []
        hidden_state = n.zeros((batch_size, self.hidden_size))
        fake_spike = self.activation(hidden_state)
        output = n.zeros((batch_size, self.output_size))
        self.initial_state = {'h': n.expand_dims(hidden_state, 1), 'z': fake_spike, 'o': output}

        for t in range(n_timestep):

            hidden_state = self.alpha * hidden_state + x[:, t, :] @ self.w_ih + fake_spike @ self.w_hh
            if self.use_bias:
                hidden_state += self.b_hh
            fake_spike = self.activation(hidden_state)
            output = self.kappa * output + fake_spike @ self.w_ho
            if self.use_readout_bias:
                output += self.b_ho
            outputs.append(output)

            if return_internals:
                hidden_states.append(hidden_state)
                fake_spikes.append(fake_spike)

        if self.return_sequence:
            return_value = n.transpose(n.stack(outputs), perm=[1, 0, 2])
        else:
            return_value = outputs[-1]

        if return_internals:
            hidden_states = [n.expand_dims(v, 0) for v in hidden_states]
            return {'h': n.transpose(n.stack(hidden_states), perm=[2, 0, 1, 3]),
                    'z': n.transpose(n.stack(fake_spikes), perm=[1, 0, 2]),
                    'output_sequence': n.transpose(n.stack(outputs), perm=[1, 0, 2]),
                    'return': return_value}
        else:
            return return_value

    def dht__dht_1(self):
        n = self.backend

        def _dht__dht_1(ht_1, *args, **kwargs):
            d = n.stack([
                n.stack([
                    n.fill(ht_1[:, 0].shape, self.alpha)
                ])])
            return n.transpose(d, perm=[2, 0, 1, 3])

        return _dht__dht_1

    def dht__dWhh(self):
        n = self.backend

        def _dht__dWhh(zt_1, *args, **kwargs):
            d = n.stack([zt_1])
            return n.transpose(d, perm=[1, 0, 2])

        return _dht__dWhh

    def dht__dWih(self):
        n = self.backend

        def _dht__dWih(xt, *args, **kwargs):
            d = n.stack([xt])
            return n.transpose(d, perm=[1, 0, 2])

        return _dht__dWih

    def dzt__dht(self):
        n = self.backend
        d_activation = self._get_d_activation()

        def _dzt__dht(ht, *args, **kwargs):
            psi = d_activation(ht[:, 0])
            return psi, n.constant([1.])

        return _dzt__dht
