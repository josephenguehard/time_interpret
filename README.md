# Time Interpret (tint)

This package expands the [captum library](https://captum.ai) with a specific 
focus on time series. Please see the documentation and examples for more details.

## Install

Time Interpret can be installed with pip:

```shell script
pip install time_interpret
```

or from source:

with pip:

```shell script
git clone git@github.com:babylonhealth/time_interpret.git
cd time_interpret
pip install -e .
```

or conda:

```shell script
git clone git@github.com:babylonhealth/time_interpret.git
cd time_interpret
conda env create
source activate tint
pip install --no-deps -e .
```


## Quick-start

First, let's load an Arma dataset:

```python
from tint.datasets import Arma

arma = Arma()
arma.download()  # This method generates the dataset
```

We then load some test data from the dataset and the
corresponding true saliency:

```python
inputs = arma.preprocess()["x"][0]
true_saliency = arma.true_saliency(dim=1)[0]
```

We can now load an attribution method and use it to compute the saliency:

```python
from tint.attr import TemporalIntegratedGradients

explainer = TemporalIntegratedGradients(arma.get_white_box)

baselines = inputs * 0
attr = explainer.attribute(
    inputs,
    baselines=baselines,
    additional_forward_args=(true_saliency,),
    temporal_additional_forward_args=(True,),
).abs()
```

Finally, we evaluate our method using the true saliency and a white box metric:

```python
from tint.metrics.white_box import aup

print(f"{aup(attr, true_saliency):.4})
```

## Methods (under development)

- [AugmentedOcclusion](https://arxiv.org/abs/2003.02821)
- [BayesLime](https://arxiv.org/pdf/2008.05030)
- [BayesShap](https://arxiv.org/pdf/2008.05030)
- [DynaMask](https://arxiv.org/pdf/2106.05303)
- [Discretised Integrated Gradients](https://arxiv.org/abs/2108.13654)
- [Fit](https://arxiv.org/abs/2003.02821)
- [Occlusion](https://arxiv.org/abs/1311.2901)
- [Retain](https://arxiv.org/pdf/1608.05745)
- [SmoothGrad](https://arxiv.org/abs/1810.03292)


## Acknowledgment
- [Jonathan Crabbe](https://github.com/JonathanCrabbe/Dynamask) for the DynaMask implementation.
- [Sana Tonekaboni](https://github.com/sanatonek/time_series_explainability/tree/master/TSX) for the fit implementation.
- [INK Lab](https://github.com/INK-USC/DIG) for the discretized integrated gradients implementation.
- [Dylan Slack](https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability) for the BayesLime and BayesShap implementations.