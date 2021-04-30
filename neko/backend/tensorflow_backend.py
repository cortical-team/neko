import numpy as np
import tensorflow as tf

# explicitly symbol for backend
N = tf


# device management
def enable_gpu():
    tf.config.set_visible_devices(tf.config.list_physical_devices())


def disable_gpu():
    tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))


def seed_random(seed):
    np.random.seed(seed)
    tf.random.set_seed(0)


float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
uint8 = tf.uint8
int8 = tf.int8
short = int16 = tf.int16
int = int32 = tf.int32
long = int64 = tf.int64
bool = tf.bool


def cast(tensor, dtype):
    return tf.cast(tensor, dtype)


# array manipulations
concat = tf.concat
expand_dims = tf.expand_dims
eye = tf.eye
fill = tf.fill
linspace = tf.linspace
ones = tf.ones
ones_like = tf.ones_like
roll = tf.roll
reshape = tf.reshape
squeeze = tf.squeeze
stack = tf.stack
unbind = tf.unstack
unsqueeze = tf.expand_dims
unstack = tf.unstack
zeros = tf.zeros
zeros_like = tf.zeros_like
where = tf.where

transpose = transpose_2d = tf.transpose


def tile(tensor, size):
    return tf.tile(tensor, size)


def slice_with_list(tensor, np_array):
    return tf.gather(tensor, np_array)


def categorical_to_onehot(tensor, n_classes):
    return tf.one_hot(tensor, n_classes)


# common single-argument element-wise math functions
abs = tf.math.abs
acos = tf.math.acos
acosh = tf.math.acosh
asin = tf.math.asin
asinh = tf.math.asinh
atan = tf.math.atan
atan2 = tf.math.atan2
atanh = tf.math.atanh
ceil = tf.math.ceil
cos = tf.math.cos
cosh = tf.math.cosh
erf = tf.math.erf
erfc = tf.math.erfc
exp = tf.math.exp
floor = tf.math.floor
log = tf.math.log
round = tf.math.round
sigmoid = tf.math.sigmoid
sign = tf.math.sign
sin = tf.math.sin
sinh = tf.math.sinh
sqrt = tf.math.sqrt
square = tf.math.square
tan = tf.math.tan
tanh = tf.math.tanh

relu = tf.nn.relu
leaky_relu = tf.nn.leaky_relu

clip_by_value = tf.clip_by_value
clamp = clip_by_value

# common multiple-argument element-wise math functions
# overloaded functions like >, etc are not included
pow = tf.math.pow
mod = tf.math.mod

# math functions with reductions
mean = tf.math.reduce_mean
reduce_mean = mean
max = tf.math.reduce_max
reduce_max = tf.math.reduce_max
min = tf.math.reduce_min
reduce_min = tf.math.reduce_min
sum = tf.math.reduce_sum
reduce_sum = tf.math.reduce_sum
prod = tf.math.reduce_prod
reduce_prod = tf.math.reduce_prod
reduce_all = tf.math.reduce_all
reduce_any = tf.math.reduce_any
reduce_std = tf.math.reduce_std
std = reduce_std

argmin = tf.math.argmin
argmax = tf.math.argmax

softmax = tf.math.softmax
log_softmax = tf.math.log_softmax
softplus = tf.math.softplus
softsign = tf.math.softsign

# other activation functions
elu = tf.keras.activations.elu
exponential = tf.keras.activations.exponential
hard_sigmoid = tf.keras.activations.hard_sigmoid
linear = tf.keras.activations.linear
selu = tf.keras.activations.selu


# other loss functions
def categorical_crossentropy(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def sparse_categorical_crossentropy(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def kullback_leibler_divergence(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.KLDivergence()(y_true, y_pred)


def mean_absolute_error(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_squared_error(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


def poisson(*, y_true, y_pred, **kwargs):
    return tf.keras.losses.Poisson()(y_true, y_pred)


kld = kullback_leibler_divergence
mae = mean_absolute_error
mse = mean_squared_error

# linear algebra operations
diag = tf.linalg.diag
diag_part = tf.linalg.diag_part
matmal = tf.linalg.matmul
matvec = tf.linalg.matvec
einsum = tf.einsum
outer = lambda x, y: tf.tensordot(x, y, axes=0)
inner = lambda x, y: tf.tensordot(x, y, axes=1)

# framework specific
is_tensor = tf.is_tensor
constant = tf.constant
tensor = tf.constant


def variable(shape, name=None):
    iter(shape)
    return tf.Variable(tf.zeros(shape=shape), name=name)


def variable_assign(tensor, data):
    if isinstance(data, np.ndarray):
        tensor.assign(tf.constant(data, dtype=tf.float32))
    elif tf.is_tensor(data):
        tensor.assign(variable_value(data))
    else:
        raise NotImplementedError()


def variable_value(tensor):
    if isinstance(tensor, tf.Variable):
        return tensor.value()
    else:
        return tensor


def variable_value_numpy(tensor):
    return tensor.numpy()


def variable_from_value(data):
    if isinstance(data, tf.Variable):
        data = data.value()
    return tf.Variable(data, dtype=tf.float32)


# random functions
def random_normal(shape, mean=0., std=1.):
    return tf.random.normal(shape=shape, mean=mean, stddev=std)


def random_uniform(shape):
    return tf.random.uniform(shape=shape)


# optimizers

def adadelta(_, learning_rate=0.001, rho=0.95, epsilon=1e-07):
    return tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon)


def adagrad(_, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07):
    return tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=initial_accumulator_value,
                                       epsilon=epsilon)


def adam(_, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                    amsgrad=amsgrad)


def adamax(_, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    return tf.keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)


def rmsprop(_, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False):
    return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, momentum=momentum, epsilon=epsilon,
                                       centered=centered)


def sgd(_, learning_rate=0.01, momentum=0.0, nesterov=False):
    return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)


def optimizer_change_parameter(optimizer, parameter_name, parameter_value):
    if parameter_name in ['learning_rate', 'lr']:
        optimizer.learning_rate.assign(parameter_value)
    else:
        raise NotImplementedError(f'Change of {parameter_name} is not supported')


# abstract operations
def calculate_gradients_and_optimize(parameters, optimizer, model_objective):
    with tf.GradientTape() as tape:
        loss = model_objective()
    optimizer.apply_gradients(zip(tape.gradient(loss, parameters), parameters))
    return loss


def loss_and_gradients(parameters, model_objective):
    with tf.GradientTape() as tape:
        loss = model_objective()
    return loss, tape.gradient(loss, parameters)


def apply_custom_gradients(parameters, gradients, optimizer):
    optimizer.apply_gradients(zip(gradients, parameters))


# symbolic derivatives
def d_sigmoid(x):
    return exp(x) / (1 + exp(x)) ** 2


def d_tanh(x):
    return 4 / (exp(x) + exp(-x)) ** 2


def d_relu(x):
    return tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))


def d_heaviside(x, v_th, gamma_pd=0.3):
    return tf.math.maximum(0., 1. - tf.math.abs(x / v_th)) * gamma_pd / v_th


def d_heaviside2(x, v_th, pseudo_bandwidth, pseudo_angle):
    ta = tf.tan(pseudo_angle)
    return tf.cast(tf.math.abs(x - v_th) < pseudo_bandwidth, tf.float32) / (2 * pseudo_bandwidth) + tf.cast(
        tf.math.abs(x - v_th) >= pseudo_bandwidth, tf.float32) * ta


# custom gradient functions
@tf.function
def heaviside(x, v_th, gamma_pd=0.3, refractory=False):
    @tf.custom_gradient
    def func(x):
        res = tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))

        def grad(dy):
            if refractory:
                return dy * tf.zeros_like(x)
            else:
                return dy * d_heaviside(x, v_th, gamma_pd)

        return res, grad

    return func(x)


@tf.function
def heaviside2(x, v_th, pseudo_bandwidth=np.float32(5e-3), pseudo_angle=np.float32(np.pi / 200), refractory=False):
    @tf.custom_gradient
    def func(x):
        res = tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))

        def grad(dy):
            if refractory:
                return dy * tf.zeros_like(x)
            else:
                return dy * d_heaviside2(x, v_th, pseudo_bandwidth=pseudo_bandwidth, pseudo_angle=pseudo_angle)

        return res, grad

    return func(x)


_derivative_registry = {sigmoid: d_sigmoid, tanh: d_tanh, relu: d_relu, heaviside: d_heaviside,
                        heaviside2: d_heaviside2}


def get_derivative(f):
    if f in _derivative_registry:
        return _derivative_registry[f]
    raise NotImplementedError(f'derivative of function {f} is not implemented.')
