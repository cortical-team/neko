import tensorflow as tf

from .base import Model
from ..backend import tensorflow_backend as tfb


class KerasAdaptor(Model):
    def __init__(self, model):
        assert isinstance(model, tf.keras.Model)
        self.model = model
        self.backend = tfb

    def parameters(self):
        return self.model.trainable_weights

    def check_or_build(self, inputs):
        pass

    def __call__(self, inputs):
        return self.model(inputs)

    def build(self, input_shape):
        pass

    def forward(self, inputs):
        return self.model(inputs)
