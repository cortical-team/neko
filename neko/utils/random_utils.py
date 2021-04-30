import numpy as np
from scipy.stats import norm, truncnorm, uniform


# random distribution from scipy
def random_normal(shape, mean, stddev, seed=None):
    return norm(loc=mean, scale=stddev).rvs(size=shape, random_state=seed).astype(np.float32)


def truncated_normal(shape, mean, stddev, seed=None):
    # tensorflow's truncated normal clips at +-2 stddev
    return truncnorm(a=-2, b=2, loc=mean, scale=stddev).rvs(size=shape, random_state=seed).astype(np.float32)


def random_uniform(shape, minval, maxval, seed=None):
    return uniform(loc=minval, scale=maxval - minval).rvs(size=shape, random_state=seed).astype(np.float32)
