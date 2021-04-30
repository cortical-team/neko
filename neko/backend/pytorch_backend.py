import numpy as np
import torch

# explicitly symbol for backend
N = torch


# device management
def enable_gpu():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def disable_gpu():
    torch.set_default_tensor_type(torch.FloatTensor)


def seed_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
uint8 = torch.uint8
int8 = torch.int8
short = int16 = torch.int16
int = int32 = torch.int32
long = int64 = torch.int64
bool = torch.bool


def cast(tensor, dtype):
    return tensor.type(dtype=dtype)


# array manipulations
concat = torch.cat
eye = torch.eye
expand_dims = torch.unsqueeze
linspace = torch.linspace
ones = torch.ones
ones_like = torch.ones_like
roll = torch.roll
reshape = torch.reshape
squeeze = torch.squeeze
stack = torch.stack
unbind = torch.unbind
unsqueeze = torch.unsqueeze
unstack = torch.unbind
zeros = torch.zeros
zeros_like = torch.zeros_like
where = torch.where

transpose_2d = lambda x: torch.transpose(x, 0, 1)


def transpose(tensor, perm):
    return tensor.permute(*perm)


def tile(tensor, size):
    return tensor.repeat(*size)


def slice_with_list(tensor, np_array):
    return tensor[np_array]


def fill(dims, value):
    t = torch.empty(dims)
    t.fill_(value)
    return t


def categorical_to_onehot(tensor, n_classes):
    return torch.eye(n_classes)[tensor]


# common single-argument element-wise math functions
abs = torch.abs
acos = torch.acos
acosh = torch.acosh
asin = torch.asin
asinh = torch.asinh
atan = torch.atan
atan2 = torch.atan2
atanh = torch.atanh
ceil = torch.ceil
cos = torch.cos
cosh = torch.cosh
erf = torch.erf
erfc = torch.erfc
exp = torch.exp
floor = torch.floor
log = torch.log
round = torch.round
sigmoid = torch.sigmoid
sign = torch.sign
sin = torch.sin
sinh = torch.sinh
sqrt = torch.sqrt
square = torch.square
tan = torch.tan
tanh = torch.tanh

relu = torch.relu


def leaky_relu(features, alpha=0.2):
    return torch.nn.LeakyReLU(alpha)(features)


clamp = torch.clamp
clip_by_value = clamp

# common multiple-argument element-wise math functions
# overloaded functions like >, etc are not included
pow = torch.pow
mod = torch.fmod

# math functions with reductions
mean = torch.mean
reduce_mean = mean
max = torch.max
reduce_max = torch.max
min = torch.min
reduce_min = torch.min
sum = torch.sum
reduce_sum = torch.sum
prod = torch.prod
reduce_prod = torch.prod
reduce_all = torch.all
reduce_any = torch.any


def std(x, unbiased=False):
    return torch.std(x, unbiased=unbiased)


reduce_std = std

argmin = torch.argmin
argmax = torch.argmax

softmax = torch.nn.Softmax(dim=-1)
log_softmax = torch.nn.LogSoftmax(dim=-1)
softplus = torch.nn.Softplus()
softsign = torch.nn.Softsign()

# other activation functions
elu = torch.nn.ELU
exponential = torch.exp
hard_sigmoid = torch.nn.Hardsigmoid()
linear = torch.nn.Identity()
selu = torch.nn.SELU()


# other loss functions
def categorical_crossentropy(*, y_true, y_pred, **kwargs):
    return torch.mean(torch.sum(-y_true * log_softmax(y_pred), dim=-1))


def sparse_categorical_crossentropy(*, y_true, y_pred, **kwargs):
    return torch.nn.CrossEntropyLoss()(y_pred, y_true)


def kullback_leibler_divergence(*, y_true, y_pred, **kwargs):
    return torch.nn.KLDivLoss()(y_pred, y_true)


def mean_absolute_error(*, y_true, y_pred, **kwargs):
    return torch.nn.L1Loss()(y_pred, y_true)


def mean_squared_error(*, y_true, y_pred, **kwargs):
    return torch.nn.MSELoss()(y_pred, y_true)


def poisson(*, y_true, y_pred, **kwargs):
    return torch.nn.PoissonNLLLoss()(y_pred, y_true)


kld = kullback_leibler_divergence
mae = mean_absolute_error
mse = mean_squared_error

# linear algebra operations
diag = torch.diag
diag_part = torch.diagonal
matmal = torch.mm
matvec = torch.mv
einsum = torch.einsum
outer = torch.ger
inner = torch.dot

# framework specific
constant = torch.tensor
tensor = torch.tensor
is_tensor = torch.is_tensor


def variable(shape, name=None):
    iter(shape)
    return torch.empty(size=shape, requires_grad=True)


def variable_assign(tensor, data):
    if isinstance(data, np.ndarray):
        tensor.data = torch.tensor(data, dtype=torch.float32)
    elif torch.is_tensor(data):
        tensor.data = data
    else:
        raise NotImplementedError()


def variable_value(tensor):
    if is_tensor(tensor):
        return torch.clone(tensor.detach()).detach()
    else:
        return tensor


def variable_value_numpy(tensor):
    arr = tensor.cpu().detach().numpy()
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.tolist()
    return arr


def variable_from_value(data):
    if is_tensor(data):
        data = data.detach()
    return torch.tensor(data, dtype=torch.float32, requires_grad=True)


# random functions
def random_normal(shape, mean=0., std=1.):
    return torch.normal(mean=mean, std=std, size=shape)


def random_uniform(shape):
    return torch.rand(*shape)


# optimizers
def adadelta(params, learning_rate=0.001, rho=0.95, epsilon=1e-07):
    return torch.optim.Adadelta(params, lr=learning_rate, rho=rho, eps=epsilon)


def adagrad(params, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07):
    return torch.optim.Adagrad(params, lr=learning_rate, initial_accumulator_value=initial_accumulator_value,
                               eps=epsilon)


def adam(params, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
    return torch.optim.Adam(params, lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon, amsgrad=amsgrad)


def adamax(params, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    return torch.optim.Adamax(params, lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon)


def rmsprop(params, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False):
    return torch.optim.RMSprop(params, lr=learning_rate, alpha=rho, momentum=momentum, eps=epsilon, centered=centered)


def sgd(params, learning_rate=0.01, momentum=0.0, nesterov=False):
    return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=nesterov)


def optimizer_change_parameter(optimizer, parameter_name, parameter_value):
    if parameter_name in ['learning_rate', 'lr']:
        param_key = 'lr'
    else:
        raise NotImplementedError(f'Change of {parameter_name} is not supported')

    for pg in optimizer.param_groups:
        pg[param_key] = parameter_value


# abstract operations
def calculate_gradients_and_optimize(_, optimizer, model_objective):
    optimizer.zero_grad()
    loss = model_objective()
    loss.backward()
    optimizer.step()
    return loss


def loss_and_gradients(parameters, model_objective):
    for p in parameters:
        if isinstance(p.grad, torch.Tensor):
            p.grad.zero_()
    loss = model_objective()
    loss.backward()
    grads = [variable_value(p.grad) for p in parameters]
    return loss, grads


def apply_custom_gradients(parameters, gradients, optimizer):
    for p in parameters:
        if p.grad is not None:
            p.grad.zero_()
    for p, g in zip(parameters, gradients):
        p.grad = g
    optimizer.step()


# symbolic derivatives
def d_sigmoid(x):
    return exp(x) / (1 + exp(x)) ** 2


def d_tanh(x):
    return 4 / (exp(x) + exp(-x)) ** 2


def d_relu(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def d_heaviside(x, v_th, gamma_pd=0.3):
    return torch.clamp(1 - torch.abs(x / v_th), min=0) * gamma_pd / v_th


def d_heaviside2(x, v_th, pseudo_bandwidth, pseudo_angle):
    ta = tan(pseudo_angle)
    return cast(abs(x - v_th) < pseudo_bandwidth, float32) / (2 * pseudo_bandwidth) + cast(
        abs(x - v_th) >= pseudo_bandwidth, float32) * ta


# custom gradient functions
class Heaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, v_th, gamma_pd, refractory):
        ctx.save_for_backward(x)
        ctx.v_th = v_th
        ctx.gamma_pd = gamma_pd
        ctx.refractory = refractory
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if ctx.refractory:
            grad_input = grad_output * torch.zeros_like(x)
        else:
            grad_input = grad_output * d_heaviside(x, ctx.v_th, ctx.gamma_pd)
        return grad_input, None, None, None


class Heaviside2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, v_th, pseudo_bandwidth, pseudo_angle, refractory):
        ctx.save_for_backward(x)
        ctx.v_th = v_th
        ctx.refractory = refractory
        ctx.pseudo_bandwidth = pseudo_bandwidth
        ctx.pseudo_angle = pseudo_angle
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if ctx.refractory:
            grad_input = grad_output * torch.zeros_like(x)
        else:
            grad_input = grad_output * d_heaviside2(x, ctx.v_th, ctx.pseudo_bandwidth, ctx.pseudo_angle)
        return grad_input, None, None, None, None


def heaviside(x, v_th, gamma_pd=0.3, refractory=False):
    return Heaviside.apply(x, v_th, gamma_pd, refractory)


def heaviside2(x, v_th, pseudo_bandwidth=np.float32(5e-3), pseudo_angle=np.float32(np.pi / 200), refractory=False):
    return Heaviside2.apply(x, v_th, pseudo_bandwidth, pseudo_angle, refractory)


_derivative_registry = {sigmoid: d_sigmoid, tanh: d_tanh, relu: d_relu, heaviside: d_heaviside,
                        heaviside2: d_heaviside2}


def get_derivative(f):
    if f in _derivative_registry:
        return _derivative_registry[f]
    raise NotImplementedError(f'derivative of function {f} is not implemented.')


# initialization script
enable_gpu()
