import optuna
import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer

from tint.attr import BayesMask
from tint.attr.models import BayesMaskNet
from tint.datasets import Arma
from tint.metrics.white_box import aup, aur, information, entropy
from tint.models import MLP


def objective(
    trial: optuna.trial.Trial,
    x: th.Tensor,
    true_saliency: th.Tensor,
    dataset: Arma,
    rare_dim: int,
    metric: str,
    accelerator: str,
    seed: int,
):
    # Create several models
    input_shape = x.shape[-1]
    model1 = MLP([input_shape, input_shape])
    model2 = MLP([input_shape, input_shape // 4, input_shape])
    model_dict = {"none": None, "model1": model1, "model2": model2}

    # Select a set of hyperparameters to test
    distribution = trial.suggest_categorical(
        "distribution", ["none", "bernoulli", "normal", "gumbel_softmax"]
    )
    hard = trial.suggest_categorical("hard", [True, False])
    model = trial.suggest_categorical("model", ["none", "model1", "model2"])
    eps = trial.suggest_float("eps", 1e-7, 1e-1, log=True)

    # Define model and trainer given the hyperparameters
    trainer = Trainer(
        max_epochs=2000,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=2,
    )
    mask = BayesMaskNet(
        forward_func=dataset.get_white_box,
        distribution=distribution,
        hard=hard,
        model=model_dict[model],
        eps=eps,
        optim="adam",
        lr=0.01,
    )

    # Log hyperparameters
    hyperparameters = dict(
        distribution=distribution,
        hard=hard,
        model=model,
        eps=eps,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    # Get attributions given the hyperparameters
    explainer = BayesMask(dataset.get_white_box)
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
    accelerator: str,
    seed: int,
    n_trials: int,
    timeout: int,
    n_jobs: int,
):
    # Load data
    arma = Arma(n_folds=5, fold=0, seed=seed)
    arma.download()

    # Only use the first 10 data points
    x = arma.preprocess()["x"][:10].to(accelerator)
    true_saliency = arma.true_saliency(dim=rare_dim)[:10].to(accelerator)

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
            rare_dim=rare_dim,
            metric=metric,
            accelerator=accelerator,
            seed=seed,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )

    trial = study.best_trial

    # Write best hyperparameters
    with open("bayes_mask_params.csv", "a") as fp:
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
        "--accelerator",
        type=str,
        default="cpu",
        help="Which accelerator to use.",
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
        default=100,
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
        accelerator=args.accelerator,
        seed=args.seed,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )
