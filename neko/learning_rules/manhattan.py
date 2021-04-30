from .base import BaseLearningRule


class ManhattanRule(BaseLearningRule):
    def __init__(self, evaluated_model, delta=1e-4, **kwargs):
        """

        Args:
            evaluated_model: A model with loss and metrics to train.
            delta: The amount of weight changed by each update
            **kwargs: Placeholder to accept and discard parameters for other algorithms.
        """
        super().__init__(evaluated_model, optimizer=None)
        self.delta = delta

    def pretraining_initialization(self):
        self.optimizer.watch(parameters=self.model.parameters())

    def step(self, x, y):
        n = self.backend

        def model_objective():
            return self.evaluated_model.loss(model=self.model, x=x, y_true=y)

        loss, gradients = self.optimizer.loss_and_gradients(model_objective)
        manhattan_gradients = [n.where(g > 0, -self.delta, self.delta) for g in gradients]
        self.optimizer.apply_gradients(self.model.parameters(), manhattan_gradients)

        if self.evaluated_model.metrics:
            return self.evaluated_model.evaluate(x, y)
        else:
            return {'loss': n.variable_value(loss)}


class Material:
    def __init__(self, delta=1e-5, gmin=1e-6, gmax=2e-3):
        self.delta = delta
        self.gmin = gmin
        self.gmax = gmax

    def potentiate(self, x, backend):
        n = backend
        return n.clamp(x + self.delta, self.gmin, self.gmax)

    def depress(self, x, backend):
        n = backend
        return n.clamp(x - self.delta, self.gmin, self.gmax)


class ManhattanMaterialRule(BaseLearningRule):
    def __init__(self, evaluated_model, material, w_factor=100, **kwargs):
        super().__init__(evaluated_model, optimizer=None)
        n = self.backend
        self.material = material
        self.w_factor = w_factor
        mean = n.mean(n.constant([self.material.gmin, self.material.gmax]))
        std = n.std(n.constant([self.material.gmin, self.material.gmax]))
        self.g = []
        for v in self.model.parameters():
            gp = n.clamp(n.random_normal(v.shape, mean, std / 100.), self.material.gmin, self.material.gmax)
            gn = n.clamp(n.random_normal(v.shape, mean, std / 100.), self.material.gmin, self.material.gmax)
            self.g.append([gp, gn])
            n.variable_assign(v, (gp - gn) * self.w_factor)

    def pretraining_initialization(self):
        self.optimizer.watch(parameters=self.model.parameters())

    def step(self, x, y):
        n = self.backend

        def model_objective():
            return self.evaluated_model.loss(model=self.model, x=x, y_true=y)

        loss, gradients = self.optimizer.loss_and_gradients(model_objective)

        trainable_parameters = self.model.parameters()
        for i in range(len(trainable_parameters)):
            self.g[i][0] = n.where(gradients[i] > 0, self.material.depress(self.g[i][0], n),
                                   self.material.potentiate(self.g[i][0], n))
            self.g[i][1] = n.where(gradients[i] > 0, self.material.potentiate(self.g[i][1], n),
                                   self.material.depress(self.g[i][1], n))
            n.variable_assign(trainable_parameters[i], (self.g[i][0] - self.g[i][1]) * self.w_factor)

        if self.evaluated_model.metrics:
            return self.evaluated_model.evaluate(x, y)
        else:
            return {'loss': n.variable_value(loss)}
