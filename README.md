# Time Interpret (tint)

[![PyPI version](https://img.shields.io/pypi/v/time_interpret.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/time_interpret)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/time_interpret.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/time_interpret)
[![PyPI downloads](https://img.shields.io/pypi/dm/time_interpret.svg)](https://pypi.org/project/time_interpret)
[![GitHub Actions status](https://github.com/josephenguehard/time_interpret/workflows/Tests/badge.svg)](https://github.com/josephenguehard/time_interpret/actions)
[![GitHub Actions status](https://github.com/josephenguehard/time_interpret/workflows/Documentation/badge.svg)](https://github.com/josephenguehard/time_interpret/actions)
[![Licence](https://img.shields.io/github/license/josephenguehard/time_interpret.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2306.02968.svg)](https://arxiv.org/abs/2306.02968)
[![Code style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)](https://github.com/psf/black)

This library expands the [captum library](https://captum.ai) with a specific 
focus on time series. For more details, please see the documentation and our [paper](https://arxiv.org/abs/2306.02968).


## Install

Time Interpret can be installed with pip:

```shell script
pip install time_interpret
```

Please see the documentation for alternative installation modes.


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

print(f"{aup(attr, true_saliency):.4}")
```

## Methods

- [AugmentedOcclusion](https://arxiv.org/abs/2003.02821)
- [BayesKernelShap](https://arxiv.org/abs/2008.05030)
- [BayesLime](https://arxiv.org/abs/2008.05030)
- [Discretized Integrated Gradients](https://arxiv.org/abs/2108.13654)
- [DynaMask](https://arxiv.org/abs/2106.05303)
- [ExtremalMask](https://arxiv.org/abs/2305.18840)
- [Fit](https://arxiv.org/abs/2003.02821)
- [LofKernelShap](https://arxiv.org/abs/2306.02968)
- [LofLime](https://arxiv.org/abs/2306.02968)
- [Non-linearities Tunnel](https://arxiv.org/abs/1906.07983)
- [Occlusion](https://arxiv.org/abs/1311.2901)
- [Retain](https://arxiv.org/abs/1608.05745)
- [SequentialIntegratedGradients](https://arxiv.org/abs/2305.15853)
- [TemporalAugmentedOcclusion](https://arxiv.org/abs/2003.02821)
- [TemporalOcclusion](https://arxiv.org/abs/2003.02821)
- [TemporalIntegratedGradients](https://arxiv.org/abs/2306.02968)
- [TimeForwardTunnel](https://arxiv.org/abs/2306.02968)

This package also provides several datasets, models and metrics. Please refer to the documentation for more details.


## Paper: Learning Perturbations to Explain Time Series Predictions

The experiments for the paper: [Learning Perturbations to Explain Time Series Predictions](https://arxiv.org/abs/2305.18840) 
can be found on these folders:
- [HMM](experiments/hmm)
- [Mimic3](experiments/mimic3/mortality)


## Paper: Sequential Integrated Gradients: a simple but effective method for explaining language models

The experiments for the paper: 
[Sequential Integrated Gradients: a simple but effective method for explaining language models](https://arxiv.org/abs/2305.15853) 
can be found on the [NLP](experiments/nlp) section of the experiments.


## TSInterpret

More methods to interpret predictions of time series classifiers have been grouped 
into [TSInterpret](https://github.com/fzi-forschungszentrum-informatik/TSInterpret), another library with a specific 
focus on time series.
We developed Time Interpret concurrently, not being aware of this library at the time.


## Acknowledgment
- [Jonathan Crabbe](https://github.com/JonathanCrabbe/Dynamask) for the DynaMask implementation.
- [Sana Tonekaboni](https://github.com/sanatonek/time_series_explainability/tree/master/TSX) for the fit implementation.
- [INK Lab](https://github.com/INK-USC/DIG) for the discretized integrated gradients' implementation.
- [Dylan Slack](https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability) for the BayesLime and BayesShap implementations.