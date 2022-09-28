import multiprocessing as mp
import random
import torch as th

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    TemporalIntegratedGradients,
    TimeForwardTunnel,
)
from tint.datasets import Hawkes
from tint.metrics.white_box import mae, mse, rmse


from experiments.hawkes.classifier import HawkesClassifierNet


def main(
    explainers: List[str],
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 0:
        device_id = [int(device.split(":")[1])]

    # Load data
    hawkes = Hawkes(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = HawkesClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=250,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    trainer.fit(classifier, datamodule=hawkes)

    # Get data for explainers
    with mp.Lock():
        x_train = hawkes.preprocess(split="train")["x"].to(device)
        x_test = hawkes.preprocess(split="test")["x"].to(device)
        y_test = hawkes.preprocess(split="test")["y"].to(device)
        true_saliency = hawkes.true_saliency(split="test").to(device)

    # Create x out of x_test and y_test
    idx = (x_test > 0).sum(1, keepdim=True)
    window = hawkes.window * th.ones_like(x_test, device=x_test.device)

    x = th.cat(
        [
            x_test,
            th.zeros_like(x_test[:, 0, :].unsqueeze(1), device=x_test.device),
        ],
        dim=1,
    )
    x.scatter_(1, idx, window)
    x = th.cat([x[:, :-1], x[:, 1:], y_test.unsqueeze(-1)], dim=-1)

    # Reshape y_test
    y_test = (
        y_test.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y_test.shape[-1], 1)
    )

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x,
                    baselines=x * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        _attr = th.cat(_attr, dim=-1).gather(-1, y_test).abs()
        _attr /= _attr.sum(2, keepdim=True)
        attr["deep_lift"] = _attr * (true_saliency.sum(2, keepdim=True) > 0)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x,
                    baselines=th.cat([x * 0, x]),
                    target=target,
                    n_samples=50,
                    stdevs=0.0001,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        _attr = th.cat(_attr, dim=-1).gather(-1, y_test).abs()
        _attr /= _attr.sum(2, keepdim=True)
        attr["gradient_shap"] = _attr * (
            true_saliency.sum(2, keepdim=True) > 0
        )

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x,
                    baselines=x * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        _attr = th.cat(_attr, dim=-1).gather(-1, y_test).abs()
        _attr /= _attr.sum(2, keepdim=True)
        attr["integrated_gradients"] = _attr * (
            true_saliency.sum(2, keepdim=True) > 0
        )

    if "temporal_integrated_gradients" in explainers:
        explainer = TemporalIntegratedGradients(classifier)
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x,
                    baselines=x * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        _attr = th.cat(_attr, dim=-1).gather(-1, y_test).abs()
        _attr /= _attr.sum(2, keepdim=True)
        attr["temporal_integrated_gradients"] = _attr * (
            true_saliency.sum(2, keepdim=True) > 0
        )

    with open("results.csv", "a") as fp, mp.Lock():
        for k, v in attr.items():
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(f"{mae(v, true_saliency, normalize=False):.4},")
            fp.write(f"{mse(v, true_saliency, normalize=False):.4},")
            fp.write(f"{rmse(v, true_saliency, normalize=False):.4}")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "gradient_shap",
            "integrated_gradients",
            "temporal_integrated_gradients",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
    )
