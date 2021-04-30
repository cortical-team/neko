"""
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2019, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2019, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2019, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2019, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from ..utils.random_utils import random_normal, random_uniform, truncated_normal


# the following initializer algorithm code from Keras 2.3.1, localized


class Initializer:
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape):
        raise NotImplementedError

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        if 'dtype' in config:
            # Initializers saved from `tf.keras`
            # may contain an unused `dtype` argument.
            config.pop('dtype')
        return cls(**config)


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape):
        return np.zeros(shape=shape, dtype=np.float32)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def __call__(self, shape):
        return np.ones(shape=shape, dtype=np.float32)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    # Arguments
        value: float; the value of the generator tensors.
    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value, dtype=np.float32)

    def get_config(self):
        return {'value': self.value}


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape):
        x = random_normal(shape, self.mean, self.stddev, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape):
        x = random_uniform(shape, self.minval, self.maxval, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x

    def get_config(self):
        return {
            'minval': self.minval,
            'maxval': self.maxval,
            'seed': self.seed,
        }


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and redrawn. This is the recommended initializer for
    neural network weights and filters.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape):
        x = truncated_normal(shape, self.mean, self.stddev, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }


class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.

    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.

    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

    def __init__(self, scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            x = truncated_normal(shape, 0., stddev, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            x = random_uniform(shape, -limit, limit, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }


class Orthogonal(Initializer):
    """Initializer that generates a random orthogonal matrix.

    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
    """

    def __init__(self, gain=1., seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        rng = np.random
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            self.seed += 1
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (self.gain * q[:shape[0], :shape[1]]).astype(np.float32)

    def get_config(self):
        return {
            'gain': self.gain,
            'seed': self.seed
        }


class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the desired matrix is not square, it gets padded
    with zeros for the additional rows/columns.

    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, shape):
        if len(shape) != 2:
            raise ValueError(
                'Identity matrix initializer '
                'can only be used for 2D matrices.')
        return (self.gain * np.eye(shape[0], shape[1])).astype(np.float32)

    def get_config(self):
        return {
            'gain': self.gain
        }


def lecun_uniform(seed=None):
    """LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)


def glorot_normal(seed=None):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='normal',
                           seed=seed)


def glorot_uniform(seed=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=seed)


def he_normal(seed=None):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    return VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def lecun_normal(seed=None):
    """LeCun normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def he_uniform(seed=None):
    """He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    return VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)


# Utility functions


def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.

    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.

    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out
