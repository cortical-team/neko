import argparse
import pickle
import time

from neko.backend import pytorch_backend as pytb
from neko.backend import tensorflow_backend as tfb
from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel, LIFRNNModel, BasicRNNModel
from neko.learning_rules import Backprop, Eprop
from neko.losses import get_loss
from neko.trainers import Trainer


def main():
    parser = argparse.ArgumentParser(description='mnist classification example')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='random seed')
    parser.add_argument('--backend', dest='backend', type=str, default='pytorch', help='choice of DL framework')
    parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='epoch to train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rule', dest='learning_rule', type=str, default='eprop', help='learning rule')
    parser.add_argument('--layer', dest='layer', type=str, default='ALIF', help='type of RNN/RSNN to use')
    parser.add_argument('--hidden', dest='hidden', type=int, default=128, help='number of neurons in a hidden layer')
    parser.add_argument('--firing_thresh', dest='firing_thresh', type=float, default=1.0, help='firing threshhold')
    parser.add_argument('--eprop_mode', dest='eprop_mode', type=str, default='adaptive', help='eprop mode to use')
    parser.add_argument('--reg', dest='reg', action='store_true', default=False, help='enable regularization')
    parser.add_argument('--reg_coeff', dest='reg_coeff', type=float, default=0.00005, help='regularization coefficient')
    parser.add_argument('--reg_target', dest='reg_target', type=int, default=10, help='regularization target')
    args = parser.parse_args()

    _layers = {'rnn': BasicRNNModel, 'lif': LIFRNNModel, 'alif': ALIFRNNModel}
    _learning_rules = {'bptt': Backprop, 'eprop': Eprop}
    _backends = {'torch': pytb, 'pytorch': pytb, 'pyt': pytb, 'tf': tfb, 'tensorflow': tfb}
    layer = _layers[args.layer.lower()]
    learning_rule = _learning_rules[args.learning_rule.lower()]
    n = _backends[args.backend.lower()]

    # generate data
    x_train, y_train, x_test, y_test = MNIST().load()
    x_train, x_test = x_train / 255., x_test / 255.

    if args.learning_rule.lower() == 'bptt' and args.reg:
        loss_fn = get_loss('categorical_crossentropy', backend=n)
        regularization_fn = get_loss('firing_rate_regularization', backend=n, firing_rate_target=args.reg_target)

        def loss_with_reg(*, model, x, y_true):
            return loss_fn(model=model, x=x, y_true=y_true) + \
                   args.reg_coeff * regularization_fn(model=model, x=x, y_true=y_true)

        loss = loss_with_reg
    else:
        loss = 'categorical_crossentropy'
    rnn = layer(args.hidden, output_size=10, backend=n, task_type='classification', return_sequence=False
                , v_th=args.firing_thresh, seed=args.seed)
    evaluated_model = Evaluator(model=rnn, loss=loss, metrics=['accuracy', 'firing_rate'])
    algo = learning_rule(evaluated_model,
                         mode=args.eprop_mode,
                         firing_rate_regularization=args.reg,
                         c_reg=args.reg_coeff,
                         f_target=args.reg_target)
    trainer = Trainer(algo)
    training_log = trainer.train(x_train, y_train, epochs=args.epoch, batch_size=args.batch_size)
    test_result = evaluated_model.evaluate(x_test, y_test, return_nparray=True)
    print('Test: ', test_result)
    completion_time = int(time.time())
    task_log = vars(args)
    task_log['name'] = 'mnist'
    task_log['log'] = training_log
    task_log['completion_time'] = completion_time
    task_log['test_result'] = test_result
    with open(f'mnist_{completion_time}.pkl', 'wb') as f:
        pickle.dump(task_log, f)


if __name__ == '__main__':
    main()
