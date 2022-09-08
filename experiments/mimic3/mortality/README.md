# Mimic III Mortality Experiments

This is the Mimic III mortality experiments section. You can use this
section to reproduce the results from the 
[DynaMask paper](https://arxiv.org/pdf/2106.05303.pdf).

To run the whole script with 5 folds, run:

```shell script
bash ./main.sh --processes 5
```

To get the experiments results on one fold only, run:

```shell script
python main.py
```

The results are saved on the csv file ``results.csv``. 

To reset the results file, run:

```shell script
python reset.py -e main
```


It is also possible to search the best hyperparameters for the BayesMask model.
To do so, run:

```shell script
python bayes_mask_params.py -p
```


## Usage

```
usage: experiments/mimci3/mortality/main.py [-h] [--explainers] [--accelerator] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["bayes_mask", "deep_lift", "dyna_mask", "fit", "gradient_shap", "integrated_gradients", "lime", "retain", "augmented_occlusion", "occlusion"]
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage : experiemnts/mimci3/mortality/main.sh [--processes] [--accelerator] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 5
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/mimci3/mortality/h_params_search.py [-h] [-p] [--metric] [--accelerator] [--seed] [--n-trials] [--timeout] [--n-jobs]

optional arguments:
  -h, --help            Show this help message and exit.
  -p, --pruning         Activate the pruning feature.
  --metric              Which metric to use as benchmark. Default to 'cross_entropy'
  --topk                Which topk to use for the metric. Default to 0.2
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Seed for train val split. Default to 42
  --n-trials            The number of trials. Default to 100
  --timeout             Stop study after the given number of second(s). Default to None
  --n-jobs              The number of parallel jobs. Default to 1
```

```
usage: experiments/mimci3/mortality/reset.py [-h] [-e]

optional arguments:
  -h, --help            Show this help message and exit.
  -e, --experiment      Name of the experiment. Either 'main' or 'bayes_mask_params'.
```
