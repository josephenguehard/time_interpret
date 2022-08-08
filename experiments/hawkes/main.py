import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from typing import List

from tint.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
    Lime,
    LOFLime,
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


from experiments.hawkes.classifier import HawkesClassifier


def main(
    explainers: List[str],
    accelerator: str = "cpu",
    fold: int = 0,
    seed: int = 42,
):
    # Load data
    hawkes = Hawkes(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = HawkesClassifier(
        feature_size=1,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(max_epochs=50, accelerator=accelerator)
    trainer.fit(classifier, datamodule=hawkes)

    # Get data for explainers
    x_train = hawkes.preprocess(split="train")["x"]
    x_test = hawkes.preprocess(split="test")["x"]
    y_test = hawkes.preprocess(split="test")["y"]

    # Switch to eval
    classifier.eval()

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier))
        attr["gradient_shap"] = explainer.attribute(
            x_test,
            baselines=th.cat([x_test * 0, x_test]),
            n_samples=50,
            stdevs=0.0001,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "lof_lime" in explainers:
        explainer = TimeForwardTunnel(LOFLime(classifier, embeddings=x_train))
        attr["lof_lime"] = explainer.attribute(
            x_test,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    if "temporal_integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(TemporalIntegratedGradients(classifier))
        attr["temporal_integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            n_steps=2,
            return_all_saliencies=True,
            show_progress=True,
        ).abs()

    # Get true saliency
    true_saliency = hawkes.true_saliency(split="test")

    with open("results.csv", "a") as fp:
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
            "lime",
            "lof_lime",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        accelerator=args.accelerator,
        fold=args.fold,
        seed=args.seed,
    )
