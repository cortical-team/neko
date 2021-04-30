import argparse
import pickle
import time

import tensorflow as tf
import torch

from neko.backend import pytorch_backend as pytb
from neko.backend import tensorflow_backend as tfb
from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import KerasAdaptor, PytorchAdaptor
from neko.learning_rules import Backprop, ManhattanRule, ManhattanMaterialRule
from neko.learning_rules.manhattan import Material
from neko.trainers import Trainer


def build_keras_model(bias=True):
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation="relu", use_bias=bias)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", use_bias=bias)(x)
    outputs = tf.keras.layers.Dense(10, activation="relu", use_bias=bias)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


class SimpleMLP(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 64, bias=bias)
        self.fc2 = torch.nn.Linear(64, 32, bias=bias)
        self.fc3 = torch.nn.Linear(32, 10, bias=bias)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        logits = torch.nn.functional.relu(self.fc3(x))
        return logits


def build_pytorch_model(bias=True):
    return SimpleMLP(bias=bias)


def main():
    parser = argparse.ArgumentParser(description='mnist classification example')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
    parser.add_argument('--framework', dest='framework', type=str, default='pytorch', help='choice of DL framework')
    parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='epoch to train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rule', dest='learning_rule', type=str, default='manhattan', help='learning rule')
    args = parser.parse_args()

    _learning_rules = {'bptt': Backprop, 'manhattan': ManhattanRule,
                       'manhattan_material': lambda x: ManhattanMaterialRule(x, material=Material())}
    learning_rule = _learning_rules[args.learning_rule.lower()]
    if args.framework.lower() in ['torch', 'pytorch', 'pyt']:
        n = pytb
        build_function = build_pytorch_model
        adaptor = PytorchAdaptor
    elif args.framework.lower() in ['tf', 'tensorflow']:
        n = tfb
        build_function = build_keras_model
        adaptor = KerasAdaptor

    if args.seed is not None:
        n.seed_random(args.seed)

    # generate data
    x_train, y_train, x_test, y_test = MNIST().load()
    x_train, x_test = x_train / 255., x_test / 255.
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    model = adaptor(build_function())
    loss = 'categorical_crossentropy'
    evaluated_model = Evaluator(model=model, loss=loss, metrics=['accuracy'])
    algo = learning_rule(evaluated_model)
    trainer = Trainer(algo)
    training_log = trainer.train(x_train, y_train, epochs=args.epoch, batch_size=args.batch_size,
                                 validation_data=(x_test, y_test))
    test_result = evaluated_model.evaluate(x_test, y_test, return_nparray=True)
    print('Test: ', test_result)
    completion_time = int(time.time())
    task_log = vars(args)
    task_log['name'] = 'mnist_manhattan'
    task_log['log'] = training_log
    task_log['completion_time'] = completion_time
    task_log['test_result'] = test_result
    with open(f'mnist_manhattan_{completion_time}.pkl', 'wb') as f:
        pickle.dump(task_log, f)


if __name__ == '__main__':
    main()
