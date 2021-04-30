import argparse
import pickle
import time

from neko.backend import pytorch_backend as pytb
from neko.backend import tensorflow_backend as tfb
from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel, LIFRNNModel, BasicRNNModel
from neko.learning_rules import Backprop, Eprop
from neko.trainers import GradientsComparisonTrainer


def get_evaluated_model(layer, hidden, backend, v_th, seed, loss):
    rnn = layer(hidden, output_size=10, backend=backend, task_type='classification', return_sequence=False
                , v_th=v_th, seed=seed)
    return Evaluator(model=rnn, loss=loss, metrics=['accuracy'])


def main():
    parser = argparse.ArgumentParser(description='mnist gradcomp example')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='random seed')
    parser.add_argument('--backend', dest='backend', type=str, default='pytorch', help='choice of DL framework')
    parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='epoch to train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--layer', dest='layer', type=str, default='ALIF', help='type of RNN/RSNN to use')
    parser.add_argument('--hidden', dest='hidden', type=int, default=128, help='number of neurons in a hidden layer')
    parser.add_argument('--firing_thresh', dest='firing_thresh', type=float, default=1.0, help='firing threshhold')
    args = parser.parse_args()

    _layers = {'rnn': BasicRNNModel, 'lif': LIFRNNModel, 'alif': ALIFRNNModel}
    _backends = {'torch': pytb, 'pytorch': pytb, 'pyt': pytb, 'tf': tfb, 'tensorflow': tfb}
    layer = _layers[args.layer.lower()]
    n = _backends[args.backend.lower()]

    # generate data
    x_train, y_train, x_test, y_test = MNIST().load()
    x_train, x_test = x_train / 255., x_test / 255.

    ev_models = [get_evaluated_model(layer, args.hidden, n, args.firing_thresh, args.seed, 'categorical_crossentropy')
                 for _ in range(4)]
    algo_ref = Backprop(ev_models[0])
    algos = [Eprop(ev_models[i], mode=mode) for i, mode in zip([1, 2, 3], ['symmetric', 'random', 'adaptive'])]
    trainer = GradientsComparisonTrainer(algo_ref, other_learning_rules=algos)
    training_log = trainer.train(x_train, y_train, epochs=args.epoch, batch_size=args.batch_size)
    grad_diffs = trainer.raw_grad_diffs
    grad_diffs_whh = []
    for algo_grad_diff in grad_diffs:
        diff_per_algo = []
        for epoch in range(args.epoch):
            diff_per_epoch = []
            # 1 is the index of whh
            for batch in range(len(algo_grad_diff[epoch])):
                diff_per_epoch.append(n.variable_value_numpy(n.mean(n.abs(algo_grad_diff[epoch][batch][1]))))
            diff_per_algo.append(diff_per_epoch)
        grad_diffs_whh.append(diff_per_algo)
    print(grad_diffs_whh)
    completion_time = int(time.time())
    task_log = vars(args)
    task_log['name'] = 'mnist_grad_comparison'
    task_log['log'] = training_log
    task_log['grad_diffs'] = grad_diffs
    with open(f'mnist_gradcomp_{completion_time}.pkl', 'wb') as f:
        pickle.dump(task_log, f)


if __name__ == '__main__':
    main()
