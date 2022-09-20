import os
import torch as th
import torchvision.transforms as T

from argparse import ArgumentParser
from captum.attr import KernelShap, Lime
from captum.metrics import sensitivity_max

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models import resnet18
from typing import List

from tint.attr import (
    AugmentedOcclusion,
    BayesLime,
    BayesKernelShap,
    LofLime,
    LofKernelShap,
    Occlusion,
)
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    lipschitz_max,
    log_odds,
    sufficiency,
)
from tint.metrics.weights import lime_weights, lof_weights
from tint.utils import get_progress_bars

from experiments.mnist.main import compute_metric


file_dir = os.path.dirname(__file__)


def main(
    explainers: List[str],
    areas: List[float],
    accelerator: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get data transform
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    target_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 255).long()),
        ]
    )

    # Load test data
    voc = VOCSegmentation(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "voc",
        ),
        image_set="val",
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    voc_loader = DataLoader(voc, batch_size=1, shuffle=True)

    # Load model
    resnet = resnet18(pretrained=True)

    # Switch to eval
    resnet.eval()

    # Set model to accelerator
    resnet.to(accelerator)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Get data as tensors
    # we only load 100 images
    x_test = list()
    seg_test = list()
    for i, (data, seg) in enumerate(voc_loader):
        if i == 100:
            break

        x_test.append(data)

        seg_ids = seg.unique()
        seg_ = seg.clone()
        for j, seg_id in enumerate(seg_ids):
            seg_[seg_ == seg_id] = j
        seg_test.append(seg_)

    x_test = th.cat(x_test).to(accelerator)
    seg_test = th.cat(seg_test).to(accelerator)

    # Target is the model prediction
    y_test = resnet(x_test).argmax(-1).to(accelerator)

    # Create dict of attributions and explainers
    attr = dict()
    expl = dict()

    if "bayes_lime" in explainers:
        explainer = BayesLime(resnet)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["bayes_lime"] = th.stack(_attr)
        expl["bayes_lime"] = explainer

    if "bayes_kernel_shap" in explainers:
        explainer = BayesKernelShap(resnet)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["bayes_kernel_shap"] = th.stack(_attr)
        expl["bayes_kernel_shap"] = explainer

    if "lime" in explainers:
        explainer = Lime(resnet)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lime"] = th.stack(_attr)
        expl["lime"] = explainer

    if "kernel_shap" in explainers:
        explainer = KernelShap(resnet)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["kernel_shap"] = th.stack(_attr)
        expl["kernel_shap"] = explainer

    if "lof_lime" in explainers:
        explainer = LofLime(resnet, embeddings=x_test)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lof_lime"] = th.stack(_attr)
        expl["lof_lime"] = explainer

    if "lof_kernel_shap" in explainers:
        explainer = LofKernelShap(resnet, embeddings=x_test)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lof_kernel_shap"] = th.stack(_attr)
        expl["lof_kernel_shap"] = explainer

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(resnet, data=x_test)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )
        expl["augmented_occlusion"] = explainer

    if "occlusion" in explainers:
        explainer = Occlusion(resnet)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )
        expl["occlusion"] = explainer

    lime_weights_fn = lime_weights(
        distance_mode="euclidean", kernel_width=1000
    )
    lof_weights_fn = lof_weights(data=x_test, n_neighbors=20)

    with open("results.csv", "a") as fp:
        for topk in get_progress_bars()(areas, desc="Topk", leave=False):
            for k, v in get_progress_bars()(
                attr.items(), desc="Attr", leave=False
            ):
                for i, weight_fn in get_progress_bars()(
                    enumerate(
                        [
                            None,
                            lime_weights_fn,
                            lof_weights_fn,
                        ]
                    ),
                    total=3,
                    desc="Weight_fn",
                    leave=False,
                ):

                    acc = accuracy(
                        resnet,
                        x_test,
                        attributions=v.cpu(),
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    comp = comprehensiveness(
                        resnet,
                        x_test,
                        attributions=v.cpu(),
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    ce = cross_entropy(
                        resnet,
                        x_test,
                        attributions=v.cpu(),
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    l_odds = log_odds(
                        resnet,
                        x_test,
                        attributions=v.cpu(),
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    suff = sufficiency(
                        resnet,
                        x_test,
                        attributions=v.cpu(),
                        topk=topk,
                        weight_fn=weight_fn,
                    )

                    fp.write(str(seed) + ",")
                    fp.write(str(topk) + ",")
                    if i == 0:
                        fp.write("None,")
                    elif i == 1:
                        fp.write("lime_weights,")
                    else:
                        fp.write("lof_weights,")
                    fp.write(k + ",")
                    fp.write(f"{acc:.4},")
                    fp.write(f"{comp:.4},")
                    fp.write(f"{ce:.4},")
                    fp.write(f"{l_odds:.4},")
                    fp.write(f"{suff:.4},")

                    for metric in get_progress_bars()(
                        [
                            accuracy,
                            comprehensiveness,
                            cross_entropy,
                            log_odds,
                            sufficiency,
                        ],
                        desc="Metric",
                        leave=False,
                    ):
                        sens_max = sensitivity_max(
                            compute_metric,
                            x_test[:10],
                            explainer=expl[k],
                            metric=metric,
                            forward_func=resnet,
                            baselines=None,
                            topk=topk,
                            target=y_test[:10],
                            additional_forward_args=seg_test[:10],
                            weight_fn=weight_fn,
                        )
                        lip_max = lipschitz_max(
                            compute_metric,
                            x_test[:10],
                            explainer=expl[k],
                            metric=metric,
                            forward_func=resnet,
                            baselines=None,
                            topk=topk,
                            target=y_test[:10],
                            additional_forward_args=seg_test[:10],
                            weight_fn=weight_fn,
                        )

                        fp.write(f"{sens_max.mean().item():4f},")
                        fp.write(f"{lip_max.mean().item():4f},")

                    fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "bayes_lime",
            "bayes_kernel_shap",
            "lime",
            "kernel_shap",
            "lof_lime",
            "lof_kernel_shap",
            "augmented_occlusion",
            "occlusion",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
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
        areas=args.areas,
        accelerator=args.accelerator,
        seed=args.seed,
        deterministic=args.deterministic,
    )
