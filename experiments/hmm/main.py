import numpy as np
import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from typing import List

from tint.attr import (
    BayesMask,
    DeepLift,
    DynaMask,
    Fit,
    GradientShap,
    IntegratedGradients,
    Lime,
    LOFLime,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalIntegratedGradients,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    BayesMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)


from experiments.hmm.classifier import StateClassifierNet


def main(
    explainers: List[str],
    accelerator: str = "cpu",
    fold: int = 0,
    seed: int = 42,
):
    # Load data
    hmm = HMM(n_folds=5, fold=fold, seed=seed)

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
    x_train = hmm.preprocess(split="train")["x"]
    x_test = hmm.preprocess(split="test")["x"]
    y_test = hmm.preprocess(split="test")["y"]

    # Switch to eval
    classifier.eval()

    # Create dict of attributions
    attr = dict()

    if "bayes_mask" in explainers:
        trainer = Trainer(max_epochs=500, accelerator=accelerator, devices=1)
        mask = BayesMaskNet(forward_func=classifier)
        explainer = BayesMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["bayes_mask"] = _attr

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            show_progress=True,
        ).abs()

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
        ).abs()

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            show_progress=True,
        ).abs()

    if "lof_lime" in explainers:
        explainer = TimeForwardTunnel(LOFLime(classifier, embeddings=x_train))
        attr["lof_lime"] = explainer.attribute(
            x_test,
            show_progress=True,
        ).abs()

    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            loss="cross_entropy",
        )
        explainer = Retain(
            datamodule=hmm,
            retain=retain,
            trainer=Trainer(max_epochs=50, accelerator=accelerator),
        )
        attr["retain"] = explainer.attribute(x_test, target=y_test).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            show_progress=True,
        ).abs()

    if "temporal_integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(TemporalIntegratedGradients(classifier))
        attr["temporal_integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            n_steps=2,
            show_progress=True,
        ).abs()

    # Get true saliency
    true_saliency = hmm.true_saliency(split="test")

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
            "bayes_mask",
            "deep_lift",
            "dyna_mask",
            "fit",
            "gradient_shap",
            "integrated_gradients",
            "lime",
            "lof_lime",
            "retain",
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
