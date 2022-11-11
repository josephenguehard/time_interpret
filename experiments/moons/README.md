# Moons experiment

This is the Moons experiment. To run the whole script with 
relu and softplus activations, on 5 different seeds run:

```shell script
bash ./main.sh
```

To get results on one seed, run:

```shell script
python main.py --seed 12
```

The results are saved in the `figures` folder on the `results.csv` file.

To reset the figures folder and the results file, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/moons/main.py [-h] [--explainers] [--n-samples] [--noises] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "geodesic_integrated_gradients", "enhanced_integrated_gradients", "gradient_shap", "integrated_gradients", "smooth_grad"]
  --n-samples           Number of samples in the dataset. Default to 10000
  --noises              List of noises to use. Default to [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  --softplus            Whether to replace relu with softplus or not. Default to False
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage : experiemnts/moons/main.sh [--processes] [--device]

optional arguments:
  --processes           Number of runners in parallel. Default to 1
  --device              Which device to use. Default to 'cpu'
```

```
usage: experiments/moons/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
