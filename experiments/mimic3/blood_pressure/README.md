# Mimic III Blood Pressure Experiments

This is the Mimic III blood pressure experiments section.

To get the experiments results on one fold only, run:

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
usage: experiments/mimci3/blood_pressure/main.py [-h] [--explainers] [--accelerator] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "gradient_shap", "integrated_gradients", "augmented_occlusion", "occlusion", "temporal_integrated_gradients"]
  --accelerator         Which accelerator to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage: experiments/mimci3/blood_pressure/reset.py [-h] [-e]

optional arguments:
  -h, --help            Show this help message and exit.
```
