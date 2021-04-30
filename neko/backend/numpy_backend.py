import numpy as np

# numpy backend only implements a subset of supported math functions, no autograd capabilities
# explicitly symbol for backend
N = np

float16 = np.float16
float32 = np.float32
float64 = np.float64
uint8 = np.uint8
int8 = np.int8
short = int16 = np.int16
int = int32 = np.int32
long = int64 = np.int64
bool = np.bool


def cast(tensor, dtype):
    return tensor.astype(dtype=dtype)


# array manipulations
concat = np.concatenate
expand_dims = np.expand_dims
eye = np.eye
fill = np.full
linspace = np.linspace
ones = np.ones
ones_like = np.ones_like
roll = np.roll
squeeze = np.squeeze
stack = np.stack
# unbind
unsqueeze = np.expand_dims
# unstack
zeros = np.zeros
zeros_like = np.zeros_like


def categorical_to_onehot(tensor, n_classes):
    return np.eye(n_classes)[tensor]


# common single-argument element-wise math functions
abs = np.abs
acos = np.arccos
acosh = np.arccosh
asin = np.arcsin
asinh = np.arcsinh
atan = np.arctan
atan2 = np.arctan2
atanh = np.arctanh
ceil = np.ceil
cos = np.cos
cosh = np.cosh
# erf
# erfc
exp = np.exp
floor = np.floor
log = np.log
round = np.round
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sign = np.sign
sin = np.sin
sinh = np.sinh
sqrt = np.sqrt
square = np.square
tan = np.tan
tanh = np.tanh

relu = lambda x: np.maximum(x, 0)
# leaky_relu

# common multiple-argument element-wise math functions
# overloaded functions like >, etc are not included
pow = np.power
mod = np.mod

# math functions with reductions
mean = np.mean
reduce_mean = mean
max = np.amax
reduce_max = np.amax
min = np.amin
reduce_min = np.amin
sum = np.sum
reduce_sum = np.sum
prod = np.prod
reduce_prod = np.prod
reduce_all = np.all
reduce_any = np.any

argmin = np.argmin
argmax = np.argmax

# linear algebra operations
diag_part = np.diag
matmal = np.matmul
matvec = np.dot
einsum = np.einsum
outer = np.outer

# framework specific
constant = np.array
is_tensor = lambda t: isinstance(t, np.ndarray)
