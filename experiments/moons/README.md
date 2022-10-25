# Moons experiment

This is the Moons experiment. To get the results, run:

```shell script
python main.py
```

The results are saved in the `figures` folder.

To reset the figures folders, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/moons/main.py [-h] [--explainers] [--n-samples] [--noises] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["geodesic_integrated_gradients", "enhanced_integrated_gradients", "gradient_shap", "integrated_gradients", "smooth_grad"]
  --n-samples           Number of samples in the dataset. Default to 10000
  --noises              List of noises to use. Default to [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage: experiments/covtype/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
