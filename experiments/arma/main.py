import numpy as np
import torch as th

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from typing import List

from tint.attr import (
    BayesMask,
    DynaMask,
    IntegratedGradients,
    Occlusion,
    FeaturePermutation,
    ShapleyValueSampling,
    TemporalIntegratedGradients,
)
from tint.attr.models import BayesMaskNet, MaskNet
from tint.datasets import Arma
from tint.metrics.white_box import aup, aur, information, entropy


def main(
    rare_dim: int,
    explainers: List[str],
    accelerator: str = "cpu",
    fold: int = 0,
    seed: int = 42,
):
    arma = Arma(n_folds=5, fold=fold, seed=seed)
    arma.download()

    # Only use the first 10 data points
    x = arma.preprocess()["x"][:10]
    true_saliency = arma.true_saliency(dim=rare_dim)[:10]

    # Create dict of attributions
    attr = dict()

    if "bayes_mask" in explainers:
        trainer = Trainer(max_epochs=500, accelerator=accelerator, devices=1)
        mask = BayesMaskNet(
            forward_func=arma.get_white_box,
            distribution="normal",
            eps=1e-1,
            optim="adam",
            lr=0.01,
        )
        explainer = BayesMask(arma.get_white_box)
        _attr = explainer.attribute(
            x,
            trainer=trainer,
            mask_net=mask,
            batch_size=50,
            additional_forward_args=(true_saliency,),
        )
        attr["bayes_mask"] = _attr

    if "dyna_mask" in explainers:
        trainer = Trainer(max_epochs=1000, accelerator=accelerator, devices=1)
        mask = MaskNet(
            forward_func=arma.get_white_box,
            perturbation="gaussian_blur",
            keep_ratio=list(np.arange(0.001, 0.051, 0.001)),
            size_reg_factor_init=1,
            size_reg_factor_dilation=1000,
            optim="adam",
            lr=0.001,
        )
        explainer = DynaMask(arma.get_white_box)
        _attr = explainer.attribute(
            x,
            trainer=trainer,
            mask_net=mask,
            batch_size=50,
            additional_forward_args=(true_saliency,),
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0]

    if "integrated_gradients" in explainers:
        attr["integrated_gradients"] = th.zeros_like(x)
        for i, (inputs, saliency) in enumerate(zip(x, true_saliency)):
            explainer = IntegratedGradients(forward_func=arma.get_white_box)
            baseline = inputs * 0
            attr["integrated_gradients"][i] = explainer.attribute(
                inputs,
                baselines=baseline,
                additional_forward_args=(saliency,),
            ).abs()

    if "occlusion" in explainers:
        attr["occlusion"] = th.zeros_like(x)
        for i, (inputs, saliency) in enumerate(zip(x, true_saliency)):
            explainer = Occlusion(forward_func=arma.get_white_box)
            baseline = th.mean(inputs, dim=0, keepdim=True)
            attr["occlusion"][i] = explainer.attribute(
                inputs,
                sliding_window_shapes=(1,),
                baselines=baseline,
                additional_forward_args=(saliency,),
            ).abs()

    if "permutation" in explainers:
        attr["permutation"] = th.zeros_like(x)
        for i, (inputs, saliency) in enumerate(zip(x, true_saliency)):
            explainer = FeaturePermutation(forward_func=arma.get_white_box)
            attr["permutation"][i] = explainer.attribute(
                inputs,
                additional_forward_args=(saliency,),
            ).abs()

    if "shapley_values_sampling" in explainers:
        attr["shapley_values_sampling"] = th.zeros_like(x)
        for i, (inputs, saliency) in enumerate(zip(x, true_saliency)):
            explainer = ShapleyValueSampling(forward_func=arma.get_white_box)
            baseline = th.mean(inputs, dim=0, keepdim=True)
            attr["shapley_values_sampling"][i] = explainer.attribute(
                inputs,
                baselines=baseline,
                additional_forward_args=(saliency,),
            ).abs()

    if "temporal_integrated_gradients" in explainers:
        attr["temporal_integrated_gradients"] = th.zeros_like(x)
        for i, (inputs, saliency) in enumerate(zip(x, true_saliency)):
            explainer = TemporalIntegratedGradients(
                forward_func=arma.get_white_box
            )
            baseline = inputs * 0
            attr["temporal_integrated_gradients"][i] = explainer.attribute(
                inputs,
                baselines=baseline,
                additional_forward_args=(saliency,),
                temporal_additional_forward_args=(True,),
            ).abs()

    with open("results.csv", "a") as fp:
        for k, v in attr.items():
            fp.write("rare-feature" if rare_dim == 1 else "rare-time")
            fp.write("," + str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4}")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--rare-dim",
        type=int,
        default=1,
        help="Whether to run the rare features or rare time experiment.",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "bayes_mask",
            "dyna_mask",
            "integrated_gradients",
            "occlusion",
            "permutation",
            "shapley_values_sampling",
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
        rare_dim=args.rare_dim,
        explainers=args.explainers,
        accelerator=args.accelerator,
        fold=args.fold,
        seed=args.seed,
    )
