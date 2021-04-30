from .base import BaseLearningRule


class Backprop(BaseLearningRule):
    def __init__(self, evaluated_model, optimizer='adam', **kwargs):
        """

        Args:
            evaluated_model: A model with loss and metrics to train.
            optimizer: Optimizer to use. Can be a string or an Optimizer object.
            **kwargs: Placeholder to accept and discard parameters for other algorithms.
        """
        super().__init__(evaluated_model, optimizer=optimizer)

    def pretraining_initialization(self):
        self.optimizer.watch(parameters=self.model.parameters())

    def step(self, x, y):
        n = self.backend

        def model_objective():
            return self.evaluated_model.loss(model=self.model, x=x, y_true=y)

        loss = self.optimizer.calculate_gradients_and_optimize(model_objective)
        if self.evaluated_model.metrics:
            return self.evaluated_model.evaluate(x, y)
        else:
            return {'loss': n.variable_value(loss)}
