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

    x_test = th.cat(x_test)
    seg_test = th.cat(seg_test)

    # Target is the model prediction
    y_test = resnet(x_test).argmax(-1)

    # Create dict of attributions
    attr = dict()

    if "bayes_lime" in explainers:
        explainer = BayesLime(resnet)
        attr["bayes_lime"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "bayes_kernel_shap" in explainers:
        explainer = BayesKernelShap(resnet)
        attr["bayes_lime"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lime" in explainers:
        explainer = Lime(resnet)
        attr["lime"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "kernel_shap" in explainers:
        explainer = KernelShap(resnet)
        attr["kernel_shap"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lof_lime" in explainers:
        explainer = LofLime(resnet, embeddings=x_test)
        attr["lof_lime"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lof_kernel_shap" in explainers:
        explainer = LofKernelShap(resnet, embeddings=x_test)
        attr["lof_kernel_shap"] = explainer.attribute(
            x_test,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(resnet, data=x_test)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    if "occlusion" in explainers:
        explainer = Occlusion(resnet)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    with open("results.csv", "a") as fp:
        for topk in areas:
            for k, v in attr.items():

                acc = accuracy(
                    resnet,
                    x_test,
                    attributions=v.cpu(),
                    topk=topk,
                )
                comp = comprehensiveness(
                    resnet,
                    x_test,
                    attributions=v.cpu(),
                    topk=topk,
                )
                ce = cross_entropy(
                    resnet,
                    x_test,
                    attributions=v.cpu(),
                    topk=topk,
                )
                l_odds = log_odds(
                    resnet,
                    x_test,
                    attributions=v.cpu(),
                    topk=topk,
                )
                suff = sufficiency(
                    resnet,
                    x_test,
                    attributions=v.cpu(),
                    topk=topk,
                )

                fp.write(str(seed) + ",")
                fp.write(str(topk) + ",")
                fp.write(k + ",")
                fp.write(f"{acc:.4},")
                fp.write(f"{comp:.4},")
                fp.write(f"{ce:.4},")
                fp.write(f"{l_odds:.4},")
                fp.write(f"{suff:.4},")
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
