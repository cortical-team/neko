from abc import abstractmethod
from math import ceil

from tqdm import tqdm

from neko.utils.training_utils import batch_generator
from ..utils.array_utils import ensure_list_of_tensor


class BaseTrainer:
    def __init__(self, learning_rule):
        self.learning_rule = learning_rule
        self.backend = self.learning_rule.backend

    @abstractmethod
    def initialize_learning_rules(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def learning_rules_step(self, x, y, epoch):
        raise NotImplementedError

    def train(self, x, y, epochs=None, batch_size=None, shuffle=True, shuffle_seed=None, validation_data=None):
        """

        Args:
            x: Input tensor.
            y: Label tensor.
            epochs: The number of times to iterate over the whole training set.
            batch_size: The batch size.
            shuffle: Whether to shuffle the training set for each epoch.
            shuffle_seed: The random seed for shuffling, no seeding if None.
            validation_data: Tuple (x, y), if present, prints validation results after each epoch

        Returns:
            A list of lists containing loss and metrics for each batch.
        """
        n = self.backend
        if not batch_size:
            batch_size = min(x.shape[0], 128)
        x, y = ensure_list_of_tensor((x, y), backend=n)
        samples = x.shape[0]
        history = {'training': []}
        if validation_data:
            history['validation'] = []
        self.initialize_learning_rules(x)

        for epoch in range(epochs):
            epoch_history = []
            batch_generator_with_progressbar = tqdm(
                enumerate(batch_generator(x, y, batch_size, shuffle, shuffle_seed, self.backend)),
                desc=f'Epoch {epoch + 1}/{epochs}',
                unit='batch',
                leave=True,
                total=ceil(samples / batch_size)
            )
            if shuffle_seed:
                shuffle_seed += 1
            for batch_idx, (batch_x, batch_y) in batch_generator_with_progressbar:
                loss_and_metrics = self.learning_rules_step(batch_x, batch_y, epoch)
                loss_and_metrics_numpy = {}
                loss_and_metrics_string = {}
                for k, v in loss_and_metrics.items():
                    v_numpy = n.variable_value_numpy(v) if n.is_tensor(v) else v
                    v_string = f'{v_numpy:.5f}'
                    loss_and_metrics_numpy[k] = v_numpy
                    loss_and_metrics_string[k] = v_string
                epoch_history.append(loss_and_metrics_numpy)
                batch_generator_with_progressbar.set_postfix(loss_and_metrics_string)
            history['training'].append(epoch_history)
            if validation_data:
                validation_result = self.learning_rule.evaluated_model.evaluate(*validation_data, return_nparray=True)
                print('validation:', validation_result)
        return history


class Trainer(BaseTrainer):
    def __init__(self, learning_rule):
        super().__init__(learning_rule)

    def initialize_learning_rules(self, inputs):
        self.learning_rule.model.check_or_build(inputs)
        self.learning_rule.pretraining_initialization()

    def learning_rules_step(self, x, y, epoch):
        return self.learning_rule.step(x, y)
