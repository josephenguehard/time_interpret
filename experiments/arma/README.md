# Arma Experiments

This is the Arma process experiments section. You can use this
section to reproduce the results from the 
[DynaMask paper](https://arxiv.org/abs/2106.05303).

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
python reset.py -e main
```


It is also possible to search the best hyperparameters for the ExtremalMask model.
To do so, run:

```shell script
python extremal_mask_params.py -p --rare-dim 1
```


## Usage

```
usage: experiments/arma/main.py [-h] [--rare-dim] [--explainers] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --rare-dim            Which type of experiment to run. 1 is for rare-feature, 2 is for rare-time. Default to 1
  --explainers          List of the explainers to use. Default to ["dyna_mask", "extremal_mask", "integrated_gradients", "occlusion", "permutation", "shapley_values_sampling"]
  --device              Which device to use. Default to 'cpu'
  --fold                Fold of the cross-validation. Default to 0
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage : experiemnts/arma/main.sh [--processes] [--device] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 10
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/arma/h_params_search.py [-h] [-p] [--rare-dim] [--metric] [--device] [--seed] [--n-trials] [--timeout] [--n-jobs]

optional arguments:
  -h, --help            Show this help message and exit.
  -p, --pruning         Activate the pruning feature.
  --rare-dim            Whether to run the rare features or rare time experiment. Default to 1
  --metric              Which metric to use as benchmark. Default to 'aur'
  --device              Which device to use. Default to 'cpu'
  --seed                Seed for train val split. Default to 42
  --n-trials            The number of trials. Default to 10
  --timeout             Stop study after the given number of second(s). Default to None
  --n-jobs              The number of parallel jobs. Default to 1
```

```
usage: experiments/arma/reset.py [-h] [-e]

optional arguments:
  -h, --help            Show this help message and exit.
  -e, --experiment      Name of the experiment. Either 'main' or 'extremal_mask_params'.
```
