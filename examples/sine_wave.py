import argparse
import pickle
import time

import numpy as np
import tensorflow as tf
import torch

from neko.backend import pytorch_backend as pytb
from neko.backend import tensorflow_backend as tfb
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel, LIFRNNModel, BasicRNNModel
from neko.learning_rules import Backprop, Eprop
from neko.losses import get_loss
from neko.trainers import Trainer


def sine_signal(seqlen=1000, size=4, clock=20, return_format='pytorch', seed=None):
    """Fangfang's sine signal generation function.

    Args:
        seqlen: Number of time steps
        size:
        clock:
        return_format: tensorflow / pytorch / numpy tensor
        seed: Random seed

    Returns:

    """
    np.random.seed(seed)
    steps = np.linspace(0, 1, seqlen + 1)
    freqs = np.array([1, 2, 3, 5])
    # freqs = np.random.randint(1, 5, size=size)
    phases = np.random.uniform(size=size)
    amps = np.random.uniform(0.5, 2, size=size)
    signals = np.array([a * np.sin(f * (steps + p) * 2 * np.pi)
                        for p, f, a in zip(phases, freqs, amps)])
    signal = signals.sum(axis=0)
    signal.resize((seqlen + 1, 1))

    if 5 <= clock <= 100:
        x = np.zeros((seqlen, clock))
        groups = 5
        period = 10
        uniques = clock // groups
        offsets = np.random.choice(10, uniques, replace=False)
        for c in range(clock):
            i, j = c // uniques, c % uniques
            index = np.arange(seqlen // groups * i, seqlen // groups * (i + 1), period) + offsets[j]
            # print(c, i, j, index)
            x[index, c] = 1
    elif clock:
        x = steps[1:].reshape(seqlen, 1)
    else:
        x = signal[:-1]

    y = signal[1:]
    if return_format in ['tf', 'tensorflow']:
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
    elif return_format in ['pytorch', 'pyt', 'torch']:
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    return steps, x, y


def sine_signal_v2(seqlen=1000, size=4, clock=20, no_noise=False, return_format='pytorch', seed=None):
    np.random.seed(seed)
    steps = np.linspace(0, 1, seqlen + 1)
    freqs = np.array([1, 2, 3, 5])
    # freqs = np.random.randint(1, 5, size=size)
    phases = np.random.uniform(size=size)
    amps = np.random.uniform(0.5, 2, size=size)
    signals = np.array([a * np.sin(f * (steps + p) * 2 * np.pi)
                        for p, f, a in zip(phases, freqs, amps)])
    signal = signals.sum(axis=0)
    signal.resize((seqlen + 1, 1))

    if not no_noise:  # based on Bellec code
        f0 = 50  # input firing rate
        input_f0 = 50 / 1000  # firing rate in KHz
        frozen_poisson_noise_input = np.random.rand(seqlen, clock) < 1. * input_f0
        x = frozen_poisson_noise_input.astype(float)
    elif 5 <= clock <= 100:
        x = np.zeros((seqlen, clock))
        groups = 5
        period = 10
        uniques = clock // groups
        offsets = np.random.choice(10, uniques, replace=False)
        for c in range(clock):
            i, j = c // uniques, c % uniques
            index = np.arange(seqlen // groups * i, seqlen // groups * (i + 1), period) + offsets[j]
            x[index, c] = 1
    elif clock:
        x = steps[1:].reshape(seqlen, 1)
    else:
        x = signal[:-1]

    y = signal[1:]
    if return_format in ['tf', 'tensorflow']:
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
    elif return_format in ['pytorch', 'pyt', 'torch']:
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    return steps, x, y


def main():
    parser = argparse.ArgumentParser(description='Learn sine wave')
    parser.add_argument('--type', dest='type', type=int, default=2, help='version of sine wave generator to use')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='random seed')
    parser.add_argument('--backend', dest='backend', type=str, default='pytorch', help='choice of DL framework')
    parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='epoch to train')
    parser.add_argument('--learning_rule', dest='learning_rule', type=str, default='eprop', help='learning rule')
    parser.add_argument('--seqlen', dest='seqlen', type=int, default=1000,
                        help='number of time step of sine wave')
    parser.add_argument('--clock', dest='clock', type=int, default=20, help='clock argument of sine wave generation')
    parser.add_argument('--size', dest='size', type=int, default=4, help='size argument of sine wave generation')
    parser.add_argument('--layer', dest='layer', type=str, default='LIF', help='type of RNN/RSNN to use')
    parser.add_argument('--hidden', dest='hidden', type=int, default=600, help='number of neurons in a hidden layer')
    parser.add_argument('--firing_thresh', dest='firing_thresh', type=float, default=0.615, help='firing threshhold')
    parser.add_argument('--eprop_mode', dest='eprop_mode', type=str, default='symmetric', help='eprop mode to use')
    parser.add_argument('--reg', dest='reg', action='store_true', default=False, help='enable regularization')
    parser.add_argument('--reg_coeff', dest='reg_coeff', type=float, default=0.00005, help='regularization coefficient')
    parser.add_argument('--reg_target', dest='reg_target', type=int, default=10, help='regularization target')
    args = parser.parse_args()

    _layers = {'rnn': BasicRNNModel, 'lif': LIFRNNModel, 'alif': ALIFRNNModel}
    _learning_rules = {'bptt': Backprop, 'eprop': Eprop}
    _backends = {'torch': pytb, 'pytorch': pytb, 'pyt': pytb, 'tf': tfb, 'tensorflow': tfb}
    _f = {1: sine_signal, 2: sine_signal_v2}[args.type]
    layer = _layers[args.layer.lower()]
    learning_rule = _learning_rules[args.learning_rule.lower()]
    n = backend = _backends[args.backend.lower()]

    # generate data
    _, x, y = _f(seqlen=args.seqlen, size=args.size, clock=args.clock, return_format=args.backend, seed=args.seed)
    x = n.expand_dims(x, 0)
    y = n.expand_dims(y, 0)

    if args.learning_rule.lower() == 'bptt' and args.reg:
        loss_fn = get_loss('mse', backend=n)
        regularization_fn = get_loss('firing_rate_regularization', backend=n, firing_rate_target=args.reg_target)

        def loss_with_reg(*, model, x, y_true):
            return loss_fn(model=model, x=x, y_true=y_true) + \
                   args.reg_coeff * regularization_fn(model=model, x=x, y_true=y_true)

        loss = loss_with_reg
    else:
        loss = 'mse'

    rnn = layer(args.hidden, v_th=args.firing_thresh, backend=backend, seed=args.seed)
    evaluated_model = Evaluator(model=rnn, loss=loss, metrics=['mse', 'firing_rate'])
    algo = learning_rule(evaluated_model,
                         mode=args.eprop_mode,
                         firing_rate_regularization=args.reg,
                         c_reg=args.reg_coeff,
                         f_target=args.reg_target)
    trainer = Trainer(algo)
    training_log = trainer.train(x, y, epochs=args.epoch)
    completion_time = int(time.time())
    task_log = vars(args)
    task_log['name'] = 'sine_wave'
    task_log['log'] = training_log
    task_log['completion_time'] = completion_time
    with open(f'sinewave_{completion_time}.pkl', 'wb') as f:
        pickle.dump(task_log, f)


if __name__ == '__main__':
    main()
