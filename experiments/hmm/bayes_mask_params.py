import optuna
import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from tint.attr import BayesMask
from tint.attr.models import BayesMaskNet
from tint.datasets import HMM
from tint.metrics.white_box import aup, aur, information, entropy
from tint.models import MLP

from experiments.hmm.classifier import StateClassifierNet


def objective(
    trial: optuna.trial.Trial,
    x_test: th.Tensor,
    true_saliency: th.Tensor,
    classifier: StateClassifierNet,
    metric: str,
    accelerator: str,
):
    # Create several models
    input_shape = x_test.shape[-1]
    model1 = MLP([input_shape, input_shape])
    model2 = MLP([input_shape, input_shape, input_shape])
    model_dict = {"none": None, "model1": model1, "model2": model2}

    # Select a set of hyperparameters to test
    distribution = trial.suggest_categorical(
        "distribution", ["none", "bernoulli", "normal", "gumbel_softmax"]
    )
    hard = trial.suggest_categorical("hard", [True, False])
    model = trial.suggest_categorical("model", ["none", "model1", "model2"])
    eps = trial.suggest_float("eps", 1e-7, 1e-1, log=True)

    # Define model and trainer given the hyperparameters
    version = trial.study.study_name + "_" + str(trial._trial_id)
    trainer = Trainer(
        max_epochs=500,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=2,
        logger=TensorBoardLogger(save_dir=".", version=version),
    )
    mask = BayesMaskNet(
        forward_func=classifier,
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
    explainer = BayesMask(classifier)
    attr = explainer.attribute(
        x_test,
        additional_forward_args=(True,),
        trainer=trainer,
        mask_net=mask,
        batch_size=100,
    ).to(accelerator)

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
    metric: str,
    accelerator: str,
    seed: int,
    n_trials: int,
    timeout: int,
    n_jobs: int,
):
    # Load data
    hmm = HMM(n_folds=5, fold=0, seed=seed)

    # Create classifier
    classifier = StateClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(max_epochs=50, accelerator=accelerator)
    trainer.fit(classifier, datamodule=hmm)

    # Get data for explainers
    x_test = hmm.preprocess(split="test")["x"].to(accelerator)

    # Get true saliency
    true_saliency = hmm.true_saliency(split="test").to(accelerator)

    # Switch to eval
    classifier.eval()

    # Set model to accelerator
    classifier.to(accelerator)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

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
            x_test=x_test,
            true_saliency=true_saliency,
            classifier=classifier,
            metric=metric,
            accelerator=accelerator,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )

    trial = study.best_trial

    # Write best hyperparameters
    with open("bayes_mask_params.csv", "a") as fp:
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
        metric=args.metric,
        accelerator=args.accelerator,
        seed=args.seed,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )
