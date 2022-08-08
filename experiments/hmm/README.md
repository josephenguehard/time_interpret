# HMM Experiments

This is the HMM experiments section. You can use this
section to reproduce the results from the 
[DynaMask paper](https://arxiv.org/pdf/2106.05303.pdf).

To get the experiments results, run:

```shell script
python main.py
```

The results are saved on the csv file ``results.csv``. 

To reset the results file, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/hmm/main.py [-h] [--explainers] [--accelerator] [--seed]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["bayes_mask", "deep_lift", "dyna_mask", "fit", "gradient_shap", "integrated_gradients", "lime", "lof_lime", "retain", "augmented_occlusion", "occlusion", "temporal_integrated_gradients"]
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/hmm/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
