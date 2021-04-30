from abc import abstractmethod

from ..optimizers import get_optimizer, NaiveOptimizer


class BaseLearningRule:
    def __init__(self, evaluated_model, optimizer=None):
        """

        Args:
            evaluated_model: A model with loss and metrics to train.
            optimizer: Optimizer to use. Can be a string or an Optimizer object.
        """
        self.optimizer = optimizer
        self.evaluated_model = evaluated_model
        self.model = evaluated_model.model
        self.backend = self.evaluated_model.backend

        if not optimizer:
            self.optimizer = NaiveOptimizer()
        elif isinstance(self.optimizer, str):
            self.optimizer = get_optimizer(self.optimizer)

    @abstractmethod
    def step(self, x, y):
        """The training code for a single batch.

        Args:
            x: Input tensor.
            y: Label tensor.

        Returns:
            A dictionary of loss and metrics.
        """
        raise NotImplementedError()

    def pretraining_initialization(self):
        """Tasks to execute right before the training loop begins, if present.

        Returns:
            None
        """
        pass
