from .base import BaseLearningRule
from ..initializers import get_initializer
from ..layers.base import Epropable
from ..layers.iaf import LIFRNNModel, ALIFRNNModel


class Eprop(BaseLearningRule):
    def __init__(self,
                 evaluated_model,
                 optimizer='adam',
                 mode='symmetric',
                 broadcast_initializer='random_normal',
                 broadcast_seed=None,
                 weight_decay_regularization=False,
                 adaptive_decay=0.01,
                 firing_rate_regularization=False,
                 c_reg=0.00005,
                 f_target=10.,
                 use_sequence_label=True,
                 **kwargs):
        """An implementation of the e-prop algorithms in "A solution to the learning dilemma for recurrent networks
            of spiking neurons" (Bellec, 2020).

        Args:
            evaluated_model: A model with loss and metrics to train.
            optimizer: Optimizer to use. Can be a string or an Optimizer object.
            mode: Specify the e-prop variant to use. 'symmetric', 'random' or 'adaptive'
            broadcast_initializer: Initializer for broadcast matrix, a string or initializer.
            broadcast_seed: Random seed for broadcast matrix initialization, no seeding if None.
            weight_decay_regularization: Whether to enable weight decay regularization for adaptive e-prop.
            adaptive_decay: The weight decay regularization coefficient.
            firing_rate_regularization: Whether to enable firing rate regularization.
            c_reg: The coefficient to control the strength of firing rate regularization.
            f_target: The target firing rate in Hz for firing rate regularization.
            use_sequence_label: Whether to use the whole target sequence as the label. Only the last step is used when disabled.
            **kwargs: Placeholder to accept and discard parameters for other algorithms.
        """
        super().__init__(evaluated_model, optimizer=optimizer)
        self.mode = mode
        self.broadcast_initializer = broadcast_initializer
        self.broadcast_seed = broadcast_seed
        if weight_decay_regularization:
            self.adaptive_decay = adaptive_decay
        else:
            self.adaptive_decay = 0.
        self.firing_rate_regularization = firing_rate_regularization
        self.c_reg = c_reg
        self.f_target = f_target
        self.model_task_type = self.model.task_type
        self.use_sequence_label = use_sequence_label

        # check eprop working conditions
        if self.mode not in ['symmetric', 'random', 'adaptive']:
            raise NotImplementedError(f'specified eprop mode {self.mode} not supported.')
        assert isinstance(self.model, Epropable), 'model must implement Epropable interface'
        assert self.model.use_readout_bias, 'model using eprop should uses readout bias'
        assert hasattr(self.model, 'alpha'), 'model using eprop should have direct hidden state connections'
        assert hasattr(self.model, 'kappa'), 'model using eprop should have output decay'

        # linear and softmax corresponds to regression and classification in paper, the learning signal
        # is derived by hand in both cases. supporting other activations requires extra work
        assert self.model_task_type in ['regression', 'classification'], f'task type {self.model_task_type} unsupported'

    def pretraining_initialization(self):
        n = self.backend

        # broadcast is transpose of W_ho, initialized to random for all three modes
        self.broadcast = n.variable(shape=(self.model.hidden_size, self.model.output_size))
        n.variable_assign(tensor=self.broadcast,
                          data=get_initializer(self.broadcast_initializer)(seed=self.broadcast_seed)(
                              shape=(self.model.hidden_size, self.model.output_size)))

        self.trainable_parameters = [self.model.w_hh, self.model.w_ih, self.model.w_ho, self.model.b_ho]
        if self.mode == 'adaptive':
            self.trainable_parameters.append(self.broadcast)
        self.optimizer.watch(parameters=self.trainable_parameters)

        # prepare the eligibility trace calculation functions, tied to the self.model's configs
        self.dht__dht_1 = self.model.dht__dht_1()
        self.dht__dWhh = self.model.dht__dWhh()
        self.dht__dWih = self.model.dht__dWih()
        self.dzt__dht = self.model.dzt__dht()

    def step(self, x, y):
        n = self.backend

        batch_size, n_timestep, input_size = x.shape
        simulation_time = self.model.simulation_interval * n_timestep

        # handle labels for classification tasks
        assert len(y.shape) > 1, 'labels must be a vector, are you using sparse labels?'
        # if the label does not have time dimension, repeat the labels for each time steps
        if len(y.shape) == 2:
            y_timestep = n.transpose(
                n.reshape(n.tile(y, [n_timestep, 1]), [n_timestep, batch_size, self.model.output_size]),
                [1, 0, 2])
        else:
            y_timestep = y

        # forward self.model, get outputs and states
        output_dict = self.model.forward(x, return_internals=True)

        # prepare for eligibility vector and trace calculation
        h = output_dict['h']
        z = output_dict['z']
        outputs = output_dict['output_sequence']
        _, _, hidden_state_dimension, hidden_size = h.shape
        eligibility_vector_Whh_t = n.zeros((batch_size, hidden_state_dimension, hidden_size))
        eligibility_vector_Wih_t = n.zeros((batch_size, hidden_state_dimension, input_size))
        lowpass_zt = n.zeros((batch_size, hidden_size))

        w_hh_grad = n.zeros_like(self.model.w_hh)
        w_ih_grad = n.zeros_like(self.model.w_ih)
        w_ho_grad = n.zeros_like(self.model.w_ho)
        b_ho_grad = n.zeros_like(self.model.b_ho)

        # calculate learning signal
        # for symmetric eprop, broadcast is bounded to w_ho
        # for random, broadcast sticks to the initial random value
        # for adaptive, broadcast updates with weight change of w_ho at the end of iteration
        if self.mode == 'symmetric':
            n.variable_assign(self.broadcast, self.model.w_ho)

        # errors for all timesteps
        if self.model_task_type == 'regression':
            error = (outputs - y_timestep) * 2
        elif self.model_task_type == 'classification':
            error = (n.softmax(outputs) - y_timestep) * self.model.output_size
        else:
            raise NotImplementedError()

        grad_scaling_factor = batch_size * n_timestep * self.model.output_size
        # L_t = (outputs[t] - y[t]) @ self.broadcast
        L = n.einsum('ho,bto->tbh', self.broadcast, error) / grad_scaling_factor

        if self.firing_rate_regularization:
            L_reg = self.c_reg * (n.einsum('btj->bj', z) / simulation_time - self.f_target) / simulation_time

        # loop through t to calculate the eligibility vector and trace
        for t in range(n_timestep):
            # learning signal L_t, vector of shape (batch_size, hidden_size)
            if self.firing_rate_regularization:
                L_t = L[t] + L_reg
            else:
                L_t = L[t]

            if t == 0:
                eligibility_vector_Whh_t = n.einsum('bijm,bjm->bim',
                                                    self.dht__dht_1(self.model.initial_state['h']),
                                                    eligibility_vector_Whh_t) + self.dht__dWhh(
                    self.model.initial_state['z'])
            else:
                eligibility_vector_Whh_t = n.einsum('bijm,bjm->bim', self.dht__dht_1(h[:, t - 1, :]),
                                                    eligibility_vector_Whh_t) + self.dht__dWhh(z[:, t - 1, :])

            psi_t, coefficient_vector = self.dzt__dht(h[:, t, :])
            eligibility_trace_Whh_t = n.einsum('i,bij->bj', coefficient_vector, eligibility_vector_Whh_t)
            eligibility_trace_Wih_t = n.einsum('i,bij->bj', coefficient_vector, eligibility_vector_Wih_t)

            # for input weights with fake eprop
            eligibility_vector_Wih_t = self.model.alpha * eligibility_vector_Wih_t + self.dht__dWih(x[:, t, :])
            # for output weights without eprop
            lowpass_zt = self.model.kappa * lowpass_zt + z[:, t, :]

            # handle cases when training is forced to use only the last label
            if not self.use_sequence_label and t != n_timestep - 1:
                error = n.zeros_like(error)

            # gradient update for t
            # outer(L*psi, eli) for Whh and Wih
            # for batch training, we take the sum along the batch axis (eliminating b in einsum)
            w_hh_grad += n.einsum('bi,bi,bj->ji', L_t, psi_t, eligibility_trace_Whh_t)
            w_ih_grad += n.einsum('bi,bi,bj->ji', L_t, psi_t, eligibility_trace_Wih_t)
            w_ho_grad += n.einsum('bo,bh->ho', error[:, t, :], lowpass_zt)
            b_ho_grad += n.einsum('bo->o', error[:, t, :])

        w_ho_grad /= grad_scaling_factor
        b_ho_grad /= grad_scaling_factor

        # there is no way of implementing i!=j in the standard eprop formula
        if isinstance(self.model, LIFRNNModel) or isinstance(self.model, ALIFRNNModel):
            w_hh_grad -= n.diag(n.diag_part(w_hh_grad))

        gradients = [w_hh_grad, w_ih_grad, w_ho_grad, b_ho_grad]
        if self.mode == 'adaptive':
            gradients.append(w_ho_grad)
        self.optimizer.apply_gradients(parameters=self.trainable_parameters, gradients=gradients)

        if self.mode == 'symmetric':
            n.variable_assign(self.broadcast, self.model.w_ho)
        if self.mode == 'adaptive':
            n.variable_assign(self.model.w_ho, (1 - self.adaptive_decay) * self.model.w_ho)
            n.variable_assign(self.broadcast, (1 - self.adaptive_decay) * self.broadcast)
        return self.evaluated_model.evaluate(x, y)
