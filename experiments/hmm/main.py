import numpy as np
import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from typing import List

from tint.attr import (
    DeepLift,
    DynaMask,
    Fit,
    GradientShap,
    IntegratedGradients,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import JointFeatureGeneratorNet, MaskNet
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)


from classifier import StateClassifierNet


def main(explainers: List[str], accelerator: str = "cpu", seed: int = 42):
    # Load data
    hmm = HMM(seed=seed)
    hmm.download()

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

    attr = dict()
    x_train = hmm.preprocess(split="train")["x"]
    x_test = hmm.preprocess(split="test")["x"]

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            show_progress=True,
        )

    if "dyna_mask" in explainers:
        trainer = Trainer(max_epochs=1000, accelerator=accelerator, devices=1)
        mask = MaskNet(
            forward_func=classifier,
            perturbation="gaussian_blur",
            keep_ratio=list(np.arange(0.25, 0.35, 0.01)),
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            time_reg_factor=1.0,
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0]

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            log_every_n_steps=10,
        )
        explainer = Fit(
            classifier,
            generator=generator,
            datamodule=hmm,
            trainer=trainer,
        )
        attr["fit"] = explainer.attribute(x_test, show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier))
        attr["gradient_shap"] = explainer.attribute(
            x_test,
            baselines=th.cat([x_test * 0, x_test]),
            n_samples=50,
            stdevs=0.0001,
            show_progress=True,
        )

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            show_progress=True,
        )

    if "retain" in explainers:
        explainer = Retain(
            datamodule=hmm,
            trainer=Trainer(max_epochs=50, accelerator=accelerator),
        )
        attr["retain"] = explainer.attribute(x_test)

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(classifier, data=x_train)
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            show_progress=True,
        )

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            baselines=x_train.mean(0, keepdim=True),
            show_progress=True,
        )

    true_saliency = hmm.true_saliency(split="test")

    with open("results.csv", "a") as fp:
        for k, v in attr.items():
            fp.write(str(seed) + ",")
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
            "dyna_mask",
            "fit",
            "gradient_shap",
            "integrated_gradients",
            "retain",
            "augmented_occlusion",
            "occlusion",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        accelerator=args.accelerator,
        seed=args.seed,
    )
