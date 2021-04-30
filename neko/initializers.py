from .external.initializers import *

# initializers from modified Keras
Initializer = Initializer
Zeros = Zeros
Ones = Ones
Constant = Constant
RandomNormal = RandomNormal
RandomUniform = RandomUniform
TruncatedNormal = TruncatedNormal
VarianceScaling = VarianceScaling
Orthogonal = Orthogonal
Identity = Identity
lecun_uniform = lecun_uniform
glorot_normal = glorot_normal
glorot_uniform = glorot_uniform
he_normal = he_normal
lecun_normal = lecun_normal
he_uniform = he_uniform

# alias
zeros = Zeros
ones = Ones
constant = Constant
random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal
LecunUniform = lecun_uniform
LecunNormal = lecun_normal
XavierNormal = xavier_normal = GlorotNormal = glorot_normal
XavierUniform = xavier_uniform = GlorotUniform = glorot_uniform
KaimingUniform = kaiming_uniform = HeUniform = he_uniform
KaimingNormal = kaiming_normal = HeNormal = he_normal

_initializer_registry = {'zeros', 'Zeros', 'ones', 'Ones', 'constant', 'Constant', 'random_uniform', 'RandomUniform',
                         'normal', 'random_normal', 'RandomNormal', 'truncated_normal', 'TruncatedNormal', 'identity',
                         'Identity', 'orthogonal', 'Orthogonal', 'LecunUniform', 'lecun_uniform', 'LecunNormal',
                         'lecun_normal', 'XavierNormal', 'xavier_normal', 'GlorotNormal', 'glorot_normal',
                         'XavierUniform', 'xavier_uniform', 'GlorotUniform', 'glorot_uniform', 'KaimingUniform',
                         'kaiming_uniform', 'HeUniform', 'he_uniform', 'KaimingNormal', 'kaiming_normal', 'HeNormal',
                         'he_normal'}


def get_initializer(name):
    if name in _initializer_registry:
        return globals()[name]
    else:
        raise Exception(f'Initializer {name} is not supported.')
