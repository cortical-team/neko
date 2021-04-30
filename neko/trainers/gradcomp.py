from .base import BaseTrainer
from ..utils.training_utils import sync_model_parameters


class GradientsComparisonTrainer(BaseTrainer):
    def __init__(self, learning_rule, other_learning_rules=None):
        super().__init__(learning_rule)
        self.other_learning_rules = other_learning_rules or []
        self.raw_grad_diffs = [dict() for _ in range(len(self.other_learning_rules))]

    def initialize_learning_rules(self, inputs):
        self.learning_rule.model.check_or_build(inputs)
        self.learning_rule.pretraining_initialization()
        for algo in self.other_learning_rules:
            algo.model.check_or_build(inputs)
            sync_model_parameters(self.learning_rule.model, algo.model)
            algo.pretraining_initialization()

    def learning_rules_step(self, x, y, epoch):

        # sync all models' parameters with the master
        for algo in self.other_learning_rules:
            sync_model_parameters(self.learning_rule.model, algo.model)

        # perform one training step for all models
        loss_and_metrics = self.learning_rule.step(x, y)
        for algo in self.other_learning_rules:
            algo.step(x, y)

        # calculate the gradients difference
        params_ref = self.learning_rule.model.parameters()
        for algo_idx, algo in enumerate(self.other_learning_rules):
            params = algo.model.parameters()
            diffs = [params[i] - params_ref[i] for i in range(len(params_ref))]
            if epoch not in self.raw_grad_diffs[algo_idx]:
                self.raw_grad_diffs[algo_idx][epoch] = []
            self.raw_grad_diffs[algo_idx][epoch].append(diffs)

        return loss_and_metrics

    @property
    def grad_diffs_reduce_batch(self, reduction='mean'):
        result = []
        reduction_fn = {'mean': lambda x: sum(x) / len(x), 'sum': lambda x: sum(x)}[reduction]
        for algo_diff_dict in self.raw_grad_diffs:
            algo_diff_mean = {}
            for epoch, algo_diff in algo_diff_dict.items():
                # transposes [n_batch * n_params] to [n_params * n_batch]
                algo_diff = list(map(list, zip(*algo_diff)))
                # take element-wise mean in the n_batch dimension
                algo_diff_mean[epoch] = [reduction_fn(param_vals) for param_vals in algo_diff]
            result.append(algo_diff_mean)
        return result
