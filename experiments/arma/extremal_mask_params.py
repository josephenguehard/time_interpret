import optuna
import torch as th
import torch.nn as nn

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Union

from tint.attr import ExtremalMask
from tint.attr.models import ExtremalMaskNet
from tint.datasets import Arma
from tint.metrics.white_box import aup, aur, information, entropy
from tint.models import MLP, RNN


def objective(
    trial: optuna.trial.Trial,
    x: th.Tensor,
    true_saliency: th.Tensor,
    dataset: Arma,
    metric: str,
    accelerator: str,
    device_id: Union[list, int],
):
    # Create several models
    input_shape = x.shape[-1]
    gru = RNN(
        input_size=input_shape,
        rnn="gru",
        hidden_size=input_shape,
    )
    _bi_gru = RNN(
        input_size=input_shape,
        rnn="gru",
        hidden_size=input_shape,
        bidirectional=True,
    )
    _bi_mlp = MLP([2 * input_shape, input_shape])
    bi_gru = nn.Sequential(_bi_gru, _bi_mlp)
    model_dict = {
        "none": None,
        "gru": gru,
        "bi_gru": bi_gru,
    }

    # Select a set of hyperparameters to test
    model = trial.suggest_categorical("model", ["none", "gru", "bi_gru"])

    # Define model and trainer given the hyperparameters
    version = trial.study.study_name + "_" + str(trial._trial_id)
    trainer = Trainer(
        max_epochs=2000,
        accelerator=accelerator,
        devices=device_id,
        log_every_n_steps=2,
        logger=TensorBoardLogger(save_dir=".", version=version),
    )
    mask = ExtremalMaskNet(
        forward_func=dataset.get_white_box,
        model=model_dict[model],
        optim="adam",
        lr=0.01,
    )

    # Log hyperparameters
    hyperparameters = dict(model=model)
    trainer.logger.log_hyperparams(hyperparameters)

    # Get attributions given the hyperparameters
    explainer = ExtremalMask(dataset.get_white_box)
    attr = explainer.attribute(
        x,
        trainer=trainer,
        mask_net=mask,
        batch_size=50,
        additional_forward_args=(true_saliency,),
    )

    # Compute the metric
    if metric == "aup":
        return aup(attr, true_saliency)
    if metric == "aur":
        return aur(attr, true_saliency)
    if metric == "information":
        return information(attr, true_saliency)
    if metric == "entropy":
        return entropy(attr, true_saliency)
    raise NotImplementedError


def main(
    pruning: bool,
    rare_dim: int,
    metric: str,
    device: str,
    seed: int,
    n_trials: int,
    timeout: int,
    n_jobs: int,
):
    # Get accelerator and device
    accelerator = device.split(":")[0]
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]
    else:
        device_id = 1

    # Load data
    arma = Arma(n_folds=5, fold=0, seed=seed)
    arma.download()

    # Only use the first 10 to 20 data points
    x = arma.preprocess()["x"][10:20].to(device)
    true_saliency = arma.true_saliency(dim=rare_dim)[10:20].to(device)

    # Set pruner
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
        if pruning
        else optuna.pruners.NopPruner()
    )

    # Define study
    direction = "minimize" if metric == "entropy" else "maximize"
    study = optuna.create_study(direction=direction, pruner=pruner)

    # Find best trial
    study.optimize(
        lambda t: objective(
            trial=t,
            x=x,
            true_saliency=true_saliency,
            dataset=arma,
            metric=metric,
            accelerator=accelerator,
            device_id=device_id,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )

    # Write results
    with open("extremal_mask_params.csv", "a") as fp:
        for trial in study.trials:
            fp.write(str(rare_dim) + ",")
            fp.write(str(trial.value) + ",")
            for value in trial.params.values():
                fp.write(str(value) + ",")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        "--rare-dim",
        type=int,
        default=1,
        help="Whether to run the rare features or rare time experiment.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="aur",
        help="Which metric to use as benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for train val split.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="The number of trials.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Stop study after the given number of second(s).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="The number of parallel jobs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        pruning=args.pruning,
        rare_dim=args.rare_dim,
        metric=args.metric,
        device=args.device,
        seed=args.seed,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )
