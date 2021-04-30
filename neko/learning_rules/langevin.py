from .base import BaseLearningRule


class Langevin(BaseLearningRule):
    """ This class is implemented by Nathan Wycoff
    """

    def __init__(self, evaluated_model, x_pred=None, optimizer='adam', sd=-1, num_burnin_steps=1000, adapt_lr=True,
                 desired_accept=0.7, adapt_power=0.51, burnin=100, **kwargs):
        """

        Args:
            evaluated_model: A model with loss and metrics to train.
            optimizer: Optimizer to use. Can be a string or an Optimizer object.
            **kwargs: Placeholder to accept and discard parameters for other algorithms.
        """
        super().__init__(evaluated_model, optimizer=optimizer)
        self.sd = sd
        self.iters = 0
        self.adapt_lr = adapt_lr
        self.num_burnin_steps = num_burnin_steps
        self.desired_accept = desired_accept
        self.adapt_power = adapt_power
        self.accept_prop = 0
        self.step_size = self.optimizer.learning_rate
        self.fresh_nll = None
        self.fresh_grads = None
        self.step_sizes = []
        if x_pred is not None:
            self.preds = []
            self.x_pred = x_pred
        else:
            self.preds = None
            self.x_pred = None
        # self.prior = tf.keras.regularizers.L2(1e-4)  # L2 Regularization Penalty
        self.burnin = burnin

    def pretraining_initialization(self):
        n = self.backend
        self.n_params = 0
        params = self.model.parameters()
        for i in range(len(params)):
            self.n_params += len(n.variable_value_numpy(params[i]).flatten())
        if self.sd < 0:
            # TODO: Nonzero
            self.sd = 0. / n.sqrt(n.cast(n.constant(self.n_params), n.float32))
        self.optimizer.watch(parameters=params)

    def get_step_size(self):
        return self.step_size

    def get_accept_prop(self):
        return self.accept_prop

    def step(self, x, y):
        verbose = False
        # import IPython; IPython.embed()
        self.iters += 1
        n = self.backend

        def model_objective():
            return self.evaluated_model.loss(model=self.model, x=x, y_true=y)
            # prior_contrib = 0
            # for param in params:
            #    prior_contrib += self.prior(param)
            # return 4000*self.evaluated_model.loss(model=self.model, x=x, y_true=y) + prior_contrib

        params = self.model.parameters()
        # if self.fresh_grads is None or self.fresh_nll is None:
        nll, grads = n.loss_and_gradients(params, model_objective)
        # else:
        # nll = self.fresh_nll
        # grads = self.fresh_grads

        # Form proposal
        wpe = [n.variable_value(var) for var in params]
        # TODO: More efficient implementation via Gaussian Dropout?
        self.optimizer.apply_gradients(params, grads)
        for i in range(len(grads)):
            n.variable_assign(params[i],
                              params[i] + n.sqrt(n.constant(2. * self.sd)) * n.random_normal(shape=wpe[i].shape))

        # Evaluate Proposal
        nll_prop, grads_prop = n.loss_and_gradients(params, model_objective)

        u = n.random_uniform(shape=(1,))
        # TODO: Get sample size.
        llr = -(nll_prop - nll)

        if verbose:
            print("Prop: %f Old: %f" % (nll_prop, nll))
            print("lu: %f lrr: %f" % (n.log(u).numpy(), llr))

        if n.log(u) < llr:
            if verbose:
                print("Accepting")
            was_accepted = True
            # self.fresh_nll = nll_prop
            # self.fres_grads = grads_prop
        else:
            if verbose:
                print("Rejecting")
            # self.fresh_nll = nll
            # self.fres_grads = grads
            was_accepted = False
            # Reset parameters to previous values.
            for i in range(len(grads)):
                n.variable_assign(params[i], wpe[i])
        self.accept_prop = self.iters / (self.iters + 1) * self.accept_prop + was_accepted / (self.iters + 1)

        # Predict if desired
        if self.preds is not None and self.iters > self.burnin:
            self.preds.append(self.model.predict(n.constant(self.x_pred)))

        # Step Size Adaptation.
        if self.adapt_lr and (self.iters < self.num_burnin_steps):
            Ht = self.desired_accept - was_accepted
            self.step_size = self.step_size * n.exp(- n.pow(n.constant(1 / (self.iters + 1)), self.adapt_power) * Ht)
            self.optimizer.change_parameter('learning_rate', self.step_size)
            if verbose:
                print("Step size: %f" % self.step_size)
            self.step_sizes.append(self.step_size)

        if self.evaluated_model.metrics:
            return self.evaluated_model.evaluate(x, y)
        else:
            return {'loss': n.variable_value(nll)}
