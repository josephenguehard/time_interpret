# Time Interpret (tint)

This package expands the [captum library](https://captum.ai) with a specific 
focus on time series. However, most of the developed methods can be used with 
any dataset. Please see the documentation and examples for more details.

## Install

Time Interpret can be installed with pip:

```shell script
pip install time_interpret
```

or from source with conda:

```shell script
git clone git@github.com:babylonhealth/time_interpret.git
cd time_interpret
conda env create
source activate tint
pip install --no-deps -e .
```


## Quick-start

First, let's load Mnist data:

```python
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

mnist = MNIST(".", download=True, transform=ToTensor())
```

Let's then create a simple Neural Network and train it on this dataset using 
Pytorch-Lightning:

```python
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from tint.models import CNN, MLP, Net

cnn = CNN(units=[1, 32, 64], kernel_size=3, pooling="max_pool_2d")
mlp = MLP(units=[7744, 128, 10], dropout=0.25, activation_final="log_softmax")
net = Net([cnn, mlp], loss="nll")

trainer = Trainer(max_epochs=10)
trainer.fit(net, DataLoader(mnist, batch_size=32))
```

We have now a trained model on Mnist. We can then use any captum or tint
interpretability method:

```python
from tint.attr import BayesShap

explainer = BayesShap(net)
data, target = mnist[0]
data = data.unsqueeze(0)
data.require_grad = True

attributions = explainer.attribute(data)
```

## Methods (under development)

- [BayesLime](https://arxiv.org/pdf/2008.05030)
- [BayesShap](https://arxiv.org/pdf/2008.05030)
- [DynaMask](https://arxiv.org/pdf/2106.05303)
- [Discretised Integrated Gradients](https://arxiv.org/abs/2108.13654)


## Acknowledgment
