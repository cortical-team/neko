import torch

from .base import Model
from ..backend import pytorch_backend as pytb


class PytorchAdaptor(Model):
    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.backend = pytb

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def check_or_build(self, inputs):
        pass

    def __call__(self, inputs):
        return self.model(inputs)

    def build(self, input_shape):
        pass

    def forward(self, inputs):
        return self.model(inputs)
