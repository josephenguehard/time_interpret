# Arma Experiments

This is the Arma process experiments section. You can use this
section to reproduce the results from the 
[DynaMask](https://arxiv.org/pdf/2106.05303.pdf).

To get the ``rare-feature`` experiments results, run:

```shell script
python main.py --rare-dim 1
```

To get the ``rare-time`` experiments results, run:

```shell script
python main.py --rare-dim 2
```

The results are saved on the csv file ``results.csv``. 

To reset the results file, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/arma/main.py [-h] [--rare-dim] [--explainers] [--accelerator]

optional arguments:
  -h, --help            Show this help message and exit.
  --rare-dim            Which type of experiment to run. 1 is for rare-feature, 2 is for rare-time. Default to 1
  --eplainers           List of the explainers to use. Default to ["occlusion", "permutation", "integrated_gradients", "shapley_values_sampling", "dyna_mask"]
  --accelerator         Which accelerator to use. Default to 'cpu'
```

```
usage: experiments/arma/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
