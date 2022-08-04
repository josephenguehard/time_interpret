import numpy as np
import torch as th

from pytorch_lightning import Trainer
from typing import List

from tint.attr import (
    Occlusion,
    FeaturePermutation,
    IntegratedGradients,
    ShapleyValueSampling,
    DynaMask,
)
from tint.attr.models import MaskNet
from tint.datasets import Arma
from tint.metrics.white_box import aup, aur, information, entropy


def main(rare_dim: int, explainers: List[str], accelerator: str = "cpu"):
    arma = Arma()
    arma.download()

    x = arma.preprocess()["x"][:10]
    true_saliency = arma.true_saliency(dim=rare_dim)[:10]

    attr = dict()

    if "occlusion" in explainers:
        attr["occlusion"] = th.zeros_like(x)
        for i, inputs, saliency in enumerate(zip(x, true_saliency)):
            explainer = Occlusion(forward_func=arma.get_white_box)
            baseline = th.mean(inputs, dim=0, keepdim=True)
            attr["occlusion"][i] = explainer.attribute(
                inputs, sliding_window_shapes=(1,), baselines=baseline
            )

    if "permutation" in explainers:
        attr["permutation"] = th.zeros_like(x)
        for i, inputs, saliency in enumerate(zip(x, true_saliency)):
            explainer = FeaturePermutation(forward_func=arma.get_white_box)
            attr["permutation"][i] = explainer.attribute(inputs)

    if "integrated_gradients" in explainers:
        attr["integrated_gradients"] = th.zeros_like(x)
        for i, inputs, saliency in enumerate(zip(x, true_saliency)):
            explainer = IntegratedGradients(forward_func=arma.get_white_box)
            baseline = inputs * 0
            attr["integrated_gradients"][i] = explainer.attribute(
                inputs, baselines=baseline
            )

    if "shapley_values_sampling" in explainers:
        attr["shapley_values_sampling"] = th.zeros_like(x)
        for i, inputs, saliency in enumerate(zip(x, true_saliency)):
            explainer = ShapleyValueSampling(forward_func=arma.get_white_box)
            baseline = th.mean(inputs, dim=0, keepdim=True)
            attr["shapley_values_sampling"][i] = explainer.attribute(
                inputs, baselines=baseline
            )

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
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0]

    with open("results.csv", "a") as fp:
        for k, v in attr.items():
            fp.write(k + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write("\n")
