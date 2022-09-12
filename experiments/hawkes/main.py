import multiprocessing as mp
import random
import torch as th

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    TemporalAugmentedOcclusion,
    TemporalIntegratedGradients,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.datasets import Hawkes
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)


from experiments.hawkes.classifier import HawkesClassifierNet


def main(
    explainers: List[str],
    accelerator: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Load data
    hawkes = Hawkes(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = HawkesClassifierNet(
        feature_size=1,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.randint(0, int(1e9)),
        ),
    )
    trainer.fit(classifier, datamodule=hawkes)

    # Get data for explainers
    with mp.Lock():
        x_train = hawkes.preprocess(split="train")["x"].to(accelerator)
        x_test = hawkes.preprocess(split="test")["x"].to(accelerator)
        y_test = hawkes.preprocess(split="test")["y"].to(accelerator)
        true_saliency = hawkes.true_saliency(split="test").to(accelerator)

    # Reshape y_test
    y_test = (
        y_test.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y_test.shape[-1], 1)
    )

    # Switch to eval
    classifier.eval()

    # Set model to accelerator
    classifier.to(accelerator)

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
                    x_test,
                    baselines=x_test * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["deep_lift"] = th.cat(_attr, dim=-1).gather(-1, y_test)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x_test,
                    baselines=th.cat([x_test * 0, x_test]),
                    target=target,
                    n_samples=50,
                    stdevs=0.0001,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["gradient_shap"] = th.cat(_attr, dim=-1).gather(-1, y_test)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x_test,
                    baselines=x_test * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["integrated_gradients"] = th.cat(_attr, dim=-1).gather(-1, y_test)

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x_test,
                    sliding_window_shapes=(1,),
                    attributions_fn=abs,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["augmented_occlusion"] = th.cat(_attr, dim=-1).gather(-1, y_test)

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x_test,
                    sliding_window_shapes=(1,),
                    baselines=x_train.mean(0, keepdim=True),
                    attributions_fn=abs,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["occlusion"] = th.cat(_attr, dim=-1).gather(-1, y_test)

    if "temporal_integrated_gradients" in explainers:
        explainer = TemporalIntegratedGradients(classifier)
        _attr = list()
        for target in [0, 1]:
            _attr.append(
                explainer.attribute(
                    x_test,
                    baselines=x_test * 0,
                    target=target,
                    return_temporal_attributions=True,
                    show_progress=True,
                ).abs()
            )
        attr["temporal_integrated_gradients"] = th.cat(_attr, dim=-1).gather(
            -1, y_test
        )

    with open("results.csv", "a") as fp, mp.Lock():
        for k, v in attr.items():
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4}")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4},")
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
            "augmented_occlusion",
            "occlusion",
            "temporal_integrated_gradients",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Which accelerator to use.",
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
        accelerator=args.accelerator,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
    )
