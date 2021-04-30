from .metrics import get_metric
from .utils.array_utils import ensure_list_of_tensor, tensors_to_nparray


class Evaluator:
    def __init__(self, model, loss=None, metrics=None):
        """

        Args:
            model: A layer instance
            loss: Loss function, can be a string or function.
            metrics: Metric function, can be a string or function
        """
        self.model = model
        self.backend = self.model.backend
        self.loss = loss
        self.metrics = []
        self.metrics_name = metrics
        self._initialize()

    def _initialize(self):
        if not self.loss:
            raise ValueError('loss must be specified')
        if isinstance(self.loss, str):
            self.loss = get_metric(self.loss, self.backend)
        if not self.metrics_name:
            self.metrics_name = []
        if not isinstance(self.metrics_name, list):
            self.metrics_name = [self.metrics_name]
        for metric_name in self.metrics_name:
            if not isinstance(metric_name, str):
                raise ValueError(f'metric specification should be a string.')
            else:
                self.metrics.append(get_metric(metric_name, backend=self.backend))

    def evaluate(self, x, y, return_nparray=False):
        """

        Args:
            x: Input tensor
            y: Label tensor
            return_nparray: Whether to return a dictionary of numpy arrays.

        Returns:
            A dictionary of tensors or numpy arrays.
        """
        n = self.backend
        x, y = ensure_list_of_tensor((x, y), backend=self.backend)
        loss_and_metrics = {'loss': n.variable_value(self.loss(model=self.model, x=x, y_true=y))}
        if self.metrics:
            for name, metric in zip(self.metrics_name, self.metrics):
                loss_and_metrics[name] = n.variable_value(metric(model=self.model, x=x, y_true=y))
        if return_nparray:
            return tensors_to_nparray(loss_and_metrics)
        else:
            return loss_and_metrics
