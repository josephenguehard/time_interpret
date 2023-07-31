# VOC experiment

This is the VOC experiments section. To get results, run:

```shell script
python main.py
```

The results are saved in the `results.csv` file.

To reset this file, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/voc/main.py [-h] [--explainers] [--areas] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "geodesic_integrated_gradients", "enhanced_integrated_gradients", "guided_integrated_gradients", "gradient_shap", "noisy_gradient_shap", "input_x_gradient", "integrated_gradients", "lime", "kernel_shap", "noise_tunnel", "augmented_occlusion", "occlusion"]
  --n-images            Number of images to use. Default to 100
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage: experiments/voc/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
