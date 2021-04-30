# Neko: a Library for Exploring Neuromorphic Learning Rules

## Installation

```bash
git clone https://github.com/water-vapor/neko.git
cd neko
pip install -e .
```

## Code Example
Train a RSNN with ALIF neurons with e-prop on MNIST:

```python
from neko.backend import pytorch_backend as backend
from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel
from neko.learning_rules import Eprop
from neko.trainers import Trainer

x_train, y_train, x_test, y_test = MNIST().load()
x_train, x_test = x_train / 255., x_test / 255.

model = ALIFRNNModel(128, 10, backend=backend, task_type='classification', return_sequence=False)
evaluated_model = Evaluator(model=model, loss='categorical_crossentropy', metrics=['accuracy', 'firing_rate'])
algo = Eprop(evaluated_model, mode='symmetric')
trainer = Trainer(algo)
trainer.train(x_train, y_train, epochs=30)
```

## Example Scripts
### Learning with e-prop
Training on the MNIST dataset with the same setting above, but more options available.
For example, you can learn with BPTT and three variations of e-prop.
```bash
python examples/mnist.py
```

Training on the TIMIT dataset. You need to place the `timit_processed` folder the same place as the script containing the processed dataset
produced by a [script](https://github.com/IGITUGraz/eligibility_propagation/blob/master/Figure_2_TIMIT/timit_processing.py) from the original authors of e-prop.
```bash
python examples/timit.py
```
### Probabilistic learning with HMC
Training on the [MNIST-1D dataset](https://github.com/greydanus/mnist1d) with HMC:
```bash
python examples/mnist_1d_hmc.py
```
### Analogue Neural Network Training with Manhattan Rule
Training on the MNIST dataset with the simple Manhattan rule or Mahattan material rule:
```bash
python examples/mnist_manhattan.py
```

### Gradient Comparison Tool
Compare the gradients from BPTT with the three varients of e-prop:
```bash
python examples/mnist_gradcompare.py
```
This is a visualization from the results of the script above.

![](https://i.ibb.co/M2XDQVY/grads.png)
