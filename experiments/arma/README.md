# Arma Experiments

This is the Arma process experiments section. You can use this
section to reproduce the results from the 
[DynaMask paper](https://arxiv.org/pdf/2106.05303.pdf).

To run the whole script with 5 folds, run:

```shell script
bash ./main.sh --processes 10
```

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
usage: experiments/arma/main.py [-h] [--rare-dim] [--explainers] [--accelerator] [--seed]

optional arguments:
  -h, --help            Show this help message and exit.
  --rare-dim            Which type of experiment to run. 1 is for rare-feature, 2 is for rare-time. Default to 1
  --explainers          List of the explainers to use. Default to ["bayes_mask", "dyna_mask", "integrated_gradients", "occlusion", "permutation", "shapley_values_sampling", "temporal_integrated_gradients"]
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage : experiemnts/main.sh [--processes] [--accelerator] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 10
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/arma/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
