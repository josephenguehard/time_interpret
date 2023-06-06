# NLP Experiments

This is the NLP experiments section. You can use this
section to reproduce the results from the [SIG paper](https://arxiv.org/abs/2305.15853).

To get the experiments results, run:

```shell script
python main.py
```

The results are saved on the csv file ``results.csv``. 

To pre-compute the knns (optional), run:

```shell script
python knn.py
```

To reset the results file, run:

```shell script
python reset.py
```

## Usage

```
usage: experiments/sst2/main.py [-h] [--explainers] [--device] [--steps] [--factor] [--strategy] [--knns_path] [--n_neighbors] [--n_jobs] [--topk] [--log_n_steps]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "discretized_integrated_gradients", "gradient_shap", "input_x_gradient", "integrated_gradients"]
  --steps               Number of steps for the DIG attributions. Default to 30
  --factor              Factor for the DIG attributions steps. Default to 0
  --strategy            The algorithm to find the next anchor point. Either 'greedy' or 'maxcount'. Default to 'greedy'
  --ref-token           Which token to choose as ref. Default to "mask"
  --knns_path           Where the knns are stored. If not provided, compute them. Default to 'knns/bert_sst2.pkl'
  --n_neighbors         Number of neighbors for the knns. Default to 500
  --n_jobs              Number of jobs for the knns. Default to 1
  --topk                Topk attributions for the metrics. Default to 1
  --log_n_steps         How often to print the metrics. Default to 2
  --device              Which device to use. Default to 'cpu'
  
```

```
usage: experiments/sst2/knn.py [-h] [--n_neighbors] [--n_jobs] [--save_path]

optional arguments:
  -h, --help            Show this help message and exit.
  --n_neighbors         Number of neighbors for the knns. Default to 500
  --n_jobs              Number of jobs for the knns. Default to 1
  --save_path           Where to store the knns. Default to 'knns'
  
```

```
usage: experiments/hmm/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
