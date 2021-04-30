# This script is developed by Nathan Wycoff
import argparse
import pickle
from functools import partial

import numpy as np
from tqdm import tqdm

from neko.activations import get_activation
from neko.backend import pytorch_backend as pytb
from neko.backend import tensorflow_backend as tfb
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel, LIFRNNModel, BasicRNNModel
from neko.learning_rules import Backprop, Eprop, Langevin
from neko.losses import get_loss
from neko.trainers import Trainer

parser = argparse.ArgumentParser(description='mnist 1d classification example')
parser.add_argument('--seed', dest='seed', type=int, default=None, help='random seed')
parser.add_argument('--backend', dest='backend', type=str, default='tensorflow', help='choice of DL framework')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='epoch to train')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--learning_rule', dest='learning_rule', type=str, default='hmc', help='learning rule')
parser.add_argument('--layer', dest='layer', type=str, default='ALIF', help='type of RNN/RSNN to use')
parser.add_argument('--hidden', dest='hidden', type=int, default=256, help='number of neurons in a hidden layer')
parser.add_argument('--firing_thresh', dest='firing_thresh', type=float, default=0.0, help='firing threshhold')
parser.add_argument('--eprop_mode', dest='eprop_mode', type=str, default='adaptive', help='eprop mode to use')
parser.add_argument('--reg', dest='reg', action='store_true', default=False, help='enable regularization')
parser.add_argument('--reg_coeff', dest='reg_coeff', type=float, default=0.00005, help='regularization coefficient')
parser.add_argument('--reg_target', dest='reg_target', type=int, default=10, help='regularization target')
parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                    default='/Users/vapor/code/mnist1d/mnist1d_data.pkl', help='path to dataset pkl file')
args = parser.parse_args()

_layers = {'rnn': BasicRNNModel, 'lif': LIFRNNModel, 'alif': ALIFRNNModel}
_learning_rules = {'bptt': Backprop, 'eprop': Eprop, 'hmc': Langevin}
_backends = {'torch': pytb, 'pytorch': pytb, 'pyt': pytb, 'tf': tfb, 'tensorflow': tfb}
layer = _layers[args.layer.lower()]
learning_rule = _learning_rules[args.learning_rule.lower()]
n = _backends[args.backend.lower()]

ranges_logbandwidth = [-2, 0]
ranges_angle = [0, np.pi / 5]
trials = 1

PSEUDO_BANDWIDTH = n.variable_from_value(5e-3)
ALPHA = n.variable_from_value(np.pi / 100)

# generate data
with open(args.dataset_path, 'rb') as f:
    data = pickle.load(f)
x_train, y_train, x_test, y_test = data['x'], data['y'], data['x_test'], data['y_test']
x_train = np.expand_dims(x_train.astype(np.float32), 1)
y_train = y_train.astype(np.float32)
x_test = np.expand_dims(x_test.astype(np.float32), 1)
y_test = y_test.astype(np.float32)
y_train = n.categorical_to_onehot(y_train, 10)
y_test = n.categorical_to_onehot(y_test, 10)

if args.learning_rule.lower() == 'bptt' and args.reg:
    loss_fn = get_loss('categorical_crossentropy', backend=n)
    regularization_fn = get_loss('firing_rate_regularization', backend=n, firing_rate_target=args.reg_target)


    def loss_with_reg(*, model, x, y_true):
        return loss_fn(model=model, x=x, y_true=y_true) + \
               args.reg_coeff * regularization_fn(model=model, x=x, y_true=y_true)


    loss = loss_with_reg
elif args.learning_rule.lower() == 'hmc':
    loss_fn = get_loss('categorical_crossentropy', backend=n)
    # prior = tf.keras.regularizers.L2(1e-4)
    prior = lambda x: 1e-4 * n.sum(n.square(x))


    def loss_with_reg(*, model, x, y_true):
        prior_contrib = 0
        for param in model.parameters():
            prior_contrib += prior(param)
        return loss_fn(model=model, x=x, y_true=y_true) + 1 / x_train.shape[0] * prior_contrib


    loss = loss_with_reg
else:
    loss = 'categorical_crossentropy'

bw = n.constant(1e-1, dtype=n.float32)
angle = n.constant(np.pi / 200, dtype=n.float32)
activation = partial(get_activation('heaviside2', n), v_th=args.firing_thresh, pseudo_bandwidth=bw, pseudo_angle=angle)
d_activation = partial(n.d_heaviside2, v_th=args.firing_thresh, pseudo_bandwidth=bw, pseudo_angle=angle)
rnn = layer(args.hidden, output_size=10, backend=n, activation=activation, d_activation=d_activation,
            task_type='classification', return_sequence=False, v_th=args.firing_thresh, seed=args.seed)
evaluated_model = Evaluator(model=rnn, loss=loss, metrics=['accuracy', 'firing_rate'])
algo = learning_rule(evaluated_model,
                     mode=args.eprop_mode,
                     firing_rate_regularization=args.reg,
                     c_reg=args.reg_coeff, x_pred=x_test, burnin=200, sd=0,
                     f_target=args.reg_target)
trainer = Trainer(algo)
training_log = trainer.train(x_train, y_train, epochs=args.epoch, batch_size=args.batch_size,
                             validation_data=(x_test, y_test))
test_result = evaluated_model.evaluate(x_test, y_test, return_nparray=True)

if args.learning_rule == 'hmc':
    print(trainer.accept_prop)

    preds = n.constant(np.array([n.variable_value_numpy(x) for x in trainer.preds]))

    Q = y_train.shape[1]
    uq = np.empty(shape=[x_test.shape[0]])
    correct = np.empty(shape=[x_test.shape[0]])
    for ni in tqdm(range(x_test.shape[0])):
        # mce = 0

        # for t in range(preds.shape[0]):
        #    p = tf.nn.softmax(preds[t,ni,:])
        #    ent = -tf.reduce_sum(p * tf.math.log(p)).numpy()
        #    mce += ent / preds.shape[0]
        #    mpreds += p / preds.shape[0]
        # mpreds = mpreds.numpy()

        # Calculate mean prediction
        mpreds = n.zeros(Q)
        for t in range(preds.shape[0]):
            p = n.softmax(preds[t, ni, :])
            mpreds += p / preds.shape[0]
        mpreds = mpreds.numpy()

        # Calculate mean (reverse) cross entropy between samples and its mean
        mce = 0
        for t in range(preds.shape[0]):
            p = n.softmax(preds[t, ni, :])
            mce += -n.sum(p * np.log(mpreds)) / preds.shape[0]

        uq[ni] = mce
        correct[ni] = np.argmax(mpreds) == np.argmax(y_test[ni, :])

    correct = correct.astype(bool)

    print("Accuracy (Posterior Mean): %f" % np.mean(correct))
    print(np.mean(uq[correct]))
    print(np.mean(uq[np.logical_not(correct)]))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.boxplot([uq[correct], uq[np.logical_not(correct)]], labels=['Correct', 'Incorrect'])
    # plt.plot(trainers.step_sizes)
    plt.title('Uncertainty vs Correctness')
    plt.ylabel("Mean Cross Entropy from Posterior Mean")
    plt.savefig("hmc_uq.pdf")
    plt.close()
