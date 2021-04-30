import math
from functools import partial

from .base import Epropable
from .recurrent import RecurrentBaseModel
from ..activations import get_activation


class LIFRNNModel(RecurrentBaseModel, Epropable):
    def __init__(self,
                 hidden_size,
                 output_size=1,
                 initializer=None,
                 activation=None,
                 d_activation=None,
                 simulation_interval=0.001,
                 recurrent_membrane_time_constant=0.02,
                 output_membrane_time_constant=0.02,
                 alpha=None,
                 kappa=0.,
                 v_th=0.615,
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
            d_activation: The derivative of activation function, overrides builtin if specified.
            simulation_interval: The time between two timesteps in seconds.
            recurrent_membrane_time_constant: Membrane time constant for recurrent (hidden) neurons.
            output_membrane_time_constant: Membrane time constant for output neurons.
            alpha: The decay factor of recurrent units, will override calculations from membrane time constant if specified.
            kappa: The decay factor of output units, will override calculations from membrane time constant if specified.
            v_th: The firing threshold for spiking neurons.
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
        n = self.backend
        if not self.activation:
            self.activation = partial(get_activation('heaviside', backend), v_th=v_th)
            self.d_activation = partial(n.d_heaviside, v_th=v_th)
        if d_activation:
            self.d_activation = d_activation
        if not alpha:
            alpha = math.exp(- simulation_interval / recurrent_membrane_time_constant)
        if not kappa:
            kappa = math.exp(- simulation_interval / output_membrane_time_constant)
        self.alpha = n.constant(alpha, dtype=n.float32)
        self.kappa = n.constant(kappa, dtype=n.float32)
        self.v_th = n.constant(v_th, dtype=n.float32)
        self.simulation_interval = n.constant(simulation_interval, dtype=n.float32)
        self.recurrent_membrane_time_constant = n.constant(recurrent_membrane_time_constant, dtype=n.float32)
        self.output_membrane_time_constant = n.constant(output_membrane_time_constant, dtype=n.float32)
        self.initial_state = None

    def forward(self, inputs, n_timestep=None, *, return_internals=False):
        n = self.backend
        x = self._process_inputs(inputs, n_timestep)
        batch_size, n_timestep, _ = x.shape

        hidden_states = []
        spikes = []
        outputs = []

        hidden_state = n.zeros((batch_size, self.hidden_size))
        spike = self.activation(hidden_state - self.v_th)
        output = spike @ self.w_ho
        self.initial_state = {'h': n.expand_dims(hidden_state, 1), 'z': spike, 'o': output}
        if self.use_readout_bias:
            output += self.b_ho

        for t in range(n_timestep):

            hidden_state = self.alpha * hidden_state + spike @ self.w_hh - n.diag_part(
                self.w_hh) * spike + x[:, t, :] @ self.w_ih - self.v_th * spike
            spike = self.activation(hidden_state - self.v_th)

            output = self.kappa * output + spike @ self.w_ho
            if self.use_readout_bias:
                output += self.b_ho

            outputs.append(output)

            if return_internals:
                hidden_states.append(hidden_state)
                spikes.append(spike)

        if self.return_sequence:
            return_value = n.transpose(n.stack(outputs), perm=[1, 0, 2])
        else:
            return_value = outputs[-1]

        if return_internals:
            hidden_states = [n.expand_dims(v, 0) for v in hidden_states]
            return {'h': n.transpose(n.stack(hidden_states), perm=[2, 0, 1, 3]),
                    'z': n.transpose(n.stack(spikes), perm=[1, 0, 2]),
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
        if not self.d_activation:
            self.d_activation = self._get_d_activation()

        def _dzt__dht(ht, *args, **kwargs):
            psi = self.d_activation(ht[:, 0] - self.v_th)
            return psi, n.constant([1.])

        return _dzt__dht


class ALIFRNNModel(RecurrentBaseModel, Epropable):
    def __init__(self,
                 hidden_size,
                 output_size=1,
                 initializer=None,
                 activation=None,
                 d_activation=None,
                 simulation_interval=0.001,
                 recurrent_membrane_time_constant=0.02,
                 output_membrane_time_constant=0.02,
                 adaptation_time_constant=2.,
                 beta=0.07,
                 v_th=0.615,
                 alpha=None,
                 kappa=0.,
                 rho=None,
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
            d_activation: The derivative of activation function, overrides builtin if specified.
            simulation_interval: The time between two timesteps in seconds.
            recurrent_membrane_time_constant: Membrane time constant for recurrent (hidden) neurons.
            output_membrane_time_constant: Membrane time constant for output neurons.
            adaptation_time_constant: "Typically chosen to be in the range of the time span of the length of the working
                memory that is a relevant for a given task".
            beta: Coefficient for firing threshold adaptation.
            v_th: The default firing threshold for spiking neurons.
            alpha: The decay factor of recurrent units, will override calculations from membrane time constant if speficied.
            kappa: The decay factor of output units, will override calculations from membrane time constant if speficied.
            rho: The decay factor of threshold adaptation, will override calculations from adaptation time constant if speficied.
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
        n = self.backend
        if not self.activation:
            self.activation = partial(get_activation('heaviside', backend), v_th=v_th)
            self.d_activation = partial(n.d_heaviside, v_th=v_th)
        if d_activation:
            self.d_activation = d_activation
        if not alpha:
            alpha = math.exp(- simulation_interval / recurrent_membrane_time_constant)
        if not kappa:
            kappa = math.exp(- simulation_interval / output_membrane_time_constant)
        if not rho:
            rho = math.exp(- simulation_interval / adaptation_time_constant)
        self.alpha = n.constant(alpha, dtype=n.float32)
        self.beta = n.constant(beta, dtype=n.float32)
        self.kappa = n.constant(kappa, dtype=n.float32)
        self.rho = n.constant(rho, dtype=n.float32)
        self.v_th = n.constant(v_th, dtype=n.float32)
        self.simulation_interval = n.constant(simulation_interval, dtype=n.float32)
        self.recurrent_membrane_time_constant = n.constant(recurrent_membrane_time_constant, dtype=n.float32)
        self.output_membrane_time_constant = n.constant(output_membrane_time_constant, dtype=n.float32)
        self.adaptation_time_constant = n.constant(adaptation_time_constant, dtype=n.float32)
        self.initial_state = None

    def forward(self, inputs, n_timestep=None, *, return_internals=False):
        n = self.backend
        x = self._process_inputs(inputs, n_timestep)
        batch_size, n_timestep, _ = x.shape

        hidden_states = []
        thresholds = []
        spikes = []
        outputs = []

        hidden_state = n.zeros((batch_size, self.hidden_size))
        threshold = n.zeros((batch_size, self.hidden_size))
        spike = self.activation(hidden_state - self.v_th - self.beta * threshold)
        output = spike @ self.w_ho
        self.initial_state = {'h': n.transpose(n.stack([hidden_state, threshold]), perm=[1, 0, 2]),
                              'z': spike,
                              'o': output}
        if self.use_readout_bias:
            output += self.b_ho

        for t in range(n_timestep):
            hidden_state = self.alpha * hidden_state + spike @ self.w_hh - n.diag_part(
                self.w_hh) * spike + x[:, t, :] @ self.w_ih - self.v_th * spike
            threshold = self.rho * threshold + spike
            spike = self.activation(hidden_state - self.v_th - self.beta * threshold)

            output = self.kappa * output + spike @ self.w_ho
            if self.use_readout_bias:
                output += self.b_ho

            outputs.append(output)

            if return_internals:
                hidden_states.append(hidden_state)
                thresholds.append(threshold)
                spikes.append(spike)

        if self.return_sequence:
            return_value = n.transpose(n.stack(outputs), perm=[1, 0, 2])
        else:
            return_value = outputs[-1]

        if return_internals:
            hidden_states = [n.stack([v, a]) for v, a in zip(hidden_states, thresholds)]
            return {'h': n.transpose(n.stack(hidden_states), perm=[2, 0, 1, 3]),
                    'z': n.transpose(n.stack(spikes), perm=[1, 0, 2]),
                    'output_sequence': n.transpose(n.stack(outputs), perm=[1, 0, 2]),
                    'return': return_value}
        else:
            return return_value

    def dht__dht_1(self):
        n = self.backend
        if not self.d_activation:
            self.d_activation = self._get_d_activation()

        def _dht__dht_1(ht_1, *args, **kwargs):
            # vt_1 = ht_1[0], at_1 = ht_1[1]
            psi = self.d_activation(ht_1[:, 0] - self.beta * ht_1[:, 1] - self.v_th)
            d = n.stack([n.stack([n.fill(psi.shape, self.alpha), n.zeros_like(psi)]),
                         n.stack([psi, self.rho - self.beta * psi])])
            return n.transpose(d, perm=[2, 0, 1, 3])

        return _dht__dht_1

    def dht__dWhh(self):
        n = self.backend

        def _dht__dWhh(zt_1, *args, **kwargs):
            d = n.stack([zt_1, n.zeros_like(zt_1)])
            return n.transpose(d, perm=[1, 0, 2])

        return _dht__dWhh

    def dht__dWih(self):
        n = self.backend

        def _dht__dWih(xt, *args, **kwargs):
            d = n.stack([xt, n.zeros_like(xt)])
            return n.transpose(d, perm=[1, 0, 2])

        return _dht__dWih

    def dzt__dht(self):
        n = self.backend
        if not self.d_activation:
            self.d_activation = self._get_d_activation()

        def _dzt__dht(ht, *args, **kwargs):
            psi = self.d_activation(ht[:, 0] - self.beta * ht[:, 1] - self.v_th)
            return psi, n.stack([n.constant(1.), -self.beta])

        return _dzt__dht
