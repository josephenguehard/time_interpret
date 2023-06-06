# HMM Experiments

This is the HMM experiments section. You can use this
section to reproduce the results 
from [Learning Perturbations to Explain Time Series Predictions](https://arxiv.org/abs/2305.18840).

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


It is also possible to search the best hyperparameters for the ExtremalMask model.
To do so, run:

```shell script
python extremal_mask_params.py -p
```


Finally, an ablation study on the lambda parameters can be run using this script:

```shell script
bash ./lambda_study.sh --processes 5
```


## Usage

```
usage: experiments/hmm/main.py [-h] [--explainers] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "dyna_mask", "extremal_mask", "fit", "gradient_shap", "integrated_gradients", "lime", "retain", "augmented_occlusion", "occlusion"]
  --device              Which device to use. Default to 'cpu'
  --fold                Fold of the cross-validation. Default to 0
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
  --lambda-1            Lambda 1 hyperparameter. Default to 1.0
  --lambda-2            Lambda 2 hyperparameter. Default to 1.0
  --output-file         Where to save the results. Default to "results.csv"
```

```
usage : experiemnts/hmm/main.sh [--processes] [--device] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 5
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage : experiemnts/hmm/lambda_study.sh [--processes] [--device] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 5
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/hmm/h_params_search.py [-h] [-p] [--metric] [--device] [--seed] [--n-trials] [--timeout] [--n-jobs]

optional arguments:
  -h, --help            Show this help message and exit.
  -p, --pruning         Activate the pruning feature.
  --metric              Which metric to use as benchmark. Default to 'aur'
  --device              Which device to use. Default to 'cpu'
  --seed                Seed for train val split. Default to 42
  --n-trials            The number of trials. Default to 10
  --timeout             Stop study after the given number of second(s). Default to None
  --n-jobs              The number of parallel jobs. Default to 1
```

```
usage: experiments/hmm/reset.py [-h] [-e]

optional arguments:
  -h, --help            Show this help message and exit.
  -e, --experiment      Name of the experiment. Either 'main', 'lambda_study' or 'extremal_mask_params'.
```
