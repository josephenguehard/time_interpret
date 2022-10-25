# CoverType experiment

This is the CoverType experiment. To get the results, run:

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
usage: experiments/covtype/main.py [-h] [--explainers] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "feature_ablation", "augmented_occlusion","geodesic_integrated_gradients", "enhanced_integrated_gradients", "gradient_shap", "input_x_gradients", "integrated_gradients", "lime", "kernel_shap", "rand", "saliency", "smooth_grad", "smooth_grad_square"]
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage: experiments/covtype/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
