import matplotlib.pyplot as plt
import numpy as np
import os
import torch as th
import warnings

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from typing import List

from captum.attr import (
    DeepLift,
    FeatureAblation,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    Lime,
    KernelShap,
    NoiseTunnel,
    Saliency,
)

from tint.attr import AugmentedOcclusion, GeodesicIntegratedGradients
from tint.metrics import accuracy, comprehensiveness, sufficiency
from tint.models import Net, MLP
from tint.utils import get_progress_bars


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def main(
    explainers: List[str],
    device: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Get data
    covtype = fetch_covtype(
        data_home=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "covtype",
        )
    )
    x, y = covtype["data"], covtype["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=seed
    )

    # Convert to tensors
    x_train = th.tensor(x_train).float()
    y_train = th.tensor(y_train).long() - 1

    x_test = th.tensor(x_test).float()
    y_test = th.tensor(y_test).long() - 1

    # Normalize
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True)

    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # Batchify
    datasets = th.utils.data.TensorDataset(x_train, y_train)
    train_iter = th.utils.data.DataLoader(
        datasets, batch_size=32, shuffle=True
    )

    # Def model
    classifier = Net(MLP([54, 100, 50, 10, 7]), lr=0.001, loss="cross_entropy")

    # Fit model
    trainer = Trainer(
        max_epochs=5,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
    )
    trainer.fit(classifier, train_dataloaders=train_iter)

    # Set model to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Set data to device
    x_train = x_train.to(device)

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Get predictions
    outputs = classifier(x_test)

    # Print accuracy
    acc = (outputs.argmax(-1) == y_test).float().mean()
    print("acc: ", acc)

    # Get 1000 random samples
    idx = np.random.choice(range(len(x_test)), 1000, replace=False)
    x_test = x_test[idx]
    y_test = y_test[idx]

    # Create dict of attr
    attr = dict()

    if "feature_ablation" in explainers:
        explainer = FeatureAblation(classifier)
        attr["feature_ablation"] = explainer.attribute(x_test, target=y_test)

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(classifier, data=x_train)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            target=y_test,
            attributions_fn=abs,
        )

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()

        for x, y in get_progress_bars()(
            zip(x_test, y_test), total=len(x_test)
        ):
            rand = (
                th.rand((50,) + x.shape).sort(dim=0).values.to(x_test.device)
            )
            x_aug = x.unsqueeze(0) * rand
            gig = GeodesicIntegratedGradients(
                classifier, data=x_aug, n_neighbors=15
            )

            _attr.append(
                gig.attribute(
                    x.unsqueeze(0),
                    target=y,
                    n_steps=5,
                    internal_batch_size=200,
                )
            )
        attr["geodesic_integrated_gradients"] = th.cat(_attr)

    if "enhanced_integrated_gradients" in explainers:
        _attr = list()

        for x, y in get_progress_bars()(
            zip(x_test, y_test), total=len(x_test)
        ):
            rand = (
                th.rand((50,) + x.shape).sort(dim=0).values.to(x_test.device)
            )
            x_aug = x.unsqueeze(0) * rand
            gig = GeodesicIntegratedGradients(
                classifier, data=x_aug, n_neighbors=15
            )

            _attr.append(
                gig.attribute(
                    x.unsqueeze(0),
                    target=y,
                    n_steps=5,
                    internal_batch_size=200,
                    distance="euclidean",
                )
            )
        attr["enhanced_integrated_gradients"] = th.cat(_attr)

    if "gradient_shap" in explainers:
        explainer = GradientShap(classifier)
        attr["gradient_shap"] = explainer.attribute(
            x_test,
            baselines=x_train,
            target=y_test,
            n_samples=50,
            stdevs=0.1,
        )

    if "input_x_gradients" in explainers:
        explainer = InputXGradient(classifier)
        attr["input_x_gradients"] = explainer.attribute(x_test, target=y_test)

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(classifier)
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            target=y_test,
            n_steps=50,
        )

    if "lime" in explainers:
        explainer = Lime(classifier)
        attr["lime"] = explainer.attribute(x_test, target=y_test)

    if "kernel_shap" in explainers:
        explainer = KernelShap(classifier)
        attr["kernel_shap"] = explainer.attribute(x_test, target=y_test)

    if "rand" in explainers:
        attr["rand"] = th.rand_like(x_test)

    if "saliency" in explainers:
        explainer = Saliency(classifier)
        attr["saliency"] = explainer.attribute(x_test, target=y_test)

    if "smooth_grad" in explainers:
        explainer = NoiseTunnel(IntegratedGradients(classifier))
        attr["smooth_grad"] = explainer.attribute(
            x_test,
            baselines=x_train,
            target=y_test,
            nt_samples=50,
            n_steps=5,
            stdevs=0.1,
            draw_baseline_from_distrib=True,
        )

    if "smooth_grad_square" in explainers:
        explainer = NoiseTunnel(IntegratedGradients(classifier))
        attr["smooth_grad_square"] = explainer.attribute(
            x_test,
            baselines=x_train,
            target=y_test,
            n_steps=5,
            nt_samples=50,
            nt_type="smoothgrad_sq",
            stdevs=0.1,
            draw_baseline_from_distrib=True,
        )

    # Eval methods
    for mode in get_progress_bars()(
        ["vanilla", "abs"],
        total=2,
        leave=False,
        desc="Mode",
    ):
        for i, baselines in get_progress_bars()(
            enumerate([0, x_train]),
            total=2,
            leave=False,
            desc="Baselines",
        ):
            baselines_name = {0: "zeros", 1: "aug"}
            for stdevs in get_progress_bars()(
                [0.0, 0.1], total=2, leave=False, desc="Noise"
            ):
                stdevs_name = str(stdevs)
                for j, target in get_progress_bars()(
                    enumerate([None, y_test]),
                    total=2,
                    leave=False,
                    desc="Target",
                ):
                    target_name = {0: "preds", 1: "true_labels"}

                    acc_comp = dict()
                    acc_suff = dict()
                    comp = dict()
                    suff = dict()

                    for k, v in get_progress_bars()(
                        attr.items(), desc="Attr", leave=False
                    ):
                        acc_comp[k] = list()
                        acc_suff[k] = list()
                        comp[k] = list()
                        suff[k] = list()

                        for topk in get_progress_bars()(
                            range(1, 54), leave=False, desc=k
                        ):
                            topk /= 54.0
                            acc_comp[k].append(
                                accuracy(
                                    classifier,
                                    x_test,
                                    attributions=v.cpu().abs()
                                    if mode == "abs"
                                    else v.cpu(),
                                    baselines=baselines,
                                    target=target,
                                    n_samples=1
                                    if isinstance(baselines, int)
                                    else 50,
                                    stdevs=stdevs,
                                    draw_baseline_from_distrib=False
                                    if isinstance(baselines, int)
                                    else True,
                                    topk=topk,
                                    mask_largest=True,
                                )
                            )
                            acc_suff[k].append(
                                accuracy(
                                    classifier,
                                    x_test,
                                    attributions=v.cpu().abs()
                                    if mode == "abs"
                                    else v.cpu(),
                                    baselines=baselines,
                                    target=target,
                                    n_samples=1
                                    if isinstance(baselines, int)
                                    else 50,
                                    stdevs=stdevs,
                                    draw_baseline_from_distrib=False
                                    if isinstance(baselines, int)
                                    else True,
                                    topk=topk,
                                    mask_largest=False,
                                )
                            )
                            comp[k].append(
                                comprehensiveness(
                                    classifier,
                                    x_test,
                                    attributions=v.cpu().abs()
                                    if mode == "abs"
                                    else v.cpu(),
                                    baselines=baselines,
                                    target=target,
                                    n_samples=1
                                    if isinstance(baselines, int)
                                    else 50,
                                    stdevs=stdevs,
                                    draw_baseline_from_distrib=False
                                    if isinstance(baselines, int)
                                    else True,
                                    topk=topk,
                                )
                            )
                            suff[k].append(
                                sufficiency(
                                    classifier,
                                    x_test,
                                    attributions=v.cpu().abs()
                                    if mode == "abs"
                                    else v.cpu(),
                                    baselines=baselines,
                                    target=target,
                                    n_samples=1
                                    if isinstance(baselines, int)
                                    else 50,
                                    stdevs=stdevs,
                                    draw_baseline_from_distrib=False
                                    if isinstance(baselines, int)
                                    else True,
                                    topk=topk,
                                )
                            )

                    for m, metric in enumerate(
                        [acc_comp, acc_suff, comp, suff]
                    ):
                        metric_name = {
                            0: "acc_comp",
                            1: "acc_suff",
                            2: "comp",
                            3: "suff",
                        }
                        plt.rcParams["axes.prop_cycle"] = plt.cycler(
                            "color", plt.cm.tab20.colors
                        )
                        for k, v in metric.items():
                            plt.plot(v, label=k)
                        plt.legend()
                        plt.savefig(
                            f"figures/{str(seed)}/{metric_name[m]}_{mode}_{baselines_name[i]}_"
                            f"{stdevs_name}_{target_name[j]}.pdf"
                        )
                        plt.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "feature_ablation",
            "augmented_occlusion",
            "geodesic_integrated_gradients",
            "enhanced_integrated_gradients",
            "gradient_shap",
            "input_x_gradients",
            "integrated_gradients",
            "lime",
            "kernel_shap",
            "rand",
            "saliency",
            "smooth_grad",
            "smooth_grad_square",
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
        seed=args.seed,
        deterministic=args.deterministic,
    )
