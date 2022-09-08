import numpy as np
import os
import torch as th
import torchvision.transforms as T

from argparse import ArgumentParser
from captum.attr import KernelShap, Lime
from captum.metrics import sensitivity_max

from pytorch_lightning import Trainer, seed_everything
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
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


from experiments.mnist.classifier import MnistClassifierNet


file_dir = os.path.dirname(__file__)


def main(
    explainers: List[str],
    areas: List[float],
    n_segments: int = 20,
    accelerator: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get data transform
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    # Load train and test data
    mnist_train = MNIST(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "mnist",
        ),
        train=True,
        transform=transform,
        download=True,
    )
    mnist_train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = MNIST(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "mnist",
        ),
        train=False,
        transform=transform,
        download=True,
    )
    mnist_test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    # Create classifier
    classifier = MnistClassifierNet(
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100, accelerator=accelerator, deterministic=deterministic
    )
    trainer.fit(
        classifier,
        train_dataloaders=mnist_train_loader,
        val_dataloaders=mnist_test_loader,
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

    # Get data, labels and compute segmentation for 1000 test images
    # using the slic function from scikit-image.
    mnist_test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)

    x_test = list()
    y_test = list()
    seg_test = list()
    for i, (x, y) in enumerate(mnist_test_loader):
        if i == 1000:
            break

        sample = np.squeeze(x.numpy().astype("double"), axis=0)
        segments = slic(
            sample.reshape(28, 28, 1),
            n_segments=n_segments,
            compactness=1,
            sigma=0.1,
        ).reshape(1, 28, 28)
        x_test.append(th.from_numpy(sample))
        y_test.append(y[0])
        seg_test.append(th.from_numpy(segments))

    x_test = th.stack(x_test).float()
    y_test = th.stack(y_test).long()
    seg_test = th.stack(seg_test)

    # Get train data for LOF and AugmentedOcclusion methods
    x_train = list()
    for data, _ in mnist_train_loader:
        x_train.append(data)
    x_train = th.cat(x_train)

    # Create dict of attributions
    attr = dict()

    # Baselines is the normalised background
    baselines = -0.4242

    if "bayes_lime" in explainers:
        explainer = BayesLime(classifier)
        attr["bayes_lime"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "bayes_kernel_shap" in explainers:
        explainer = BayesKernelShap(classifier)
        attr["bayes_lime"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lime" in explainers:
        explainer = Lime(classifier)
        attr["lime"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "kernel_shap" in explainers:
        explainer = KernelShap(classifier)
        attr["kernel_shap"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lof_lime" in explainers:
        explainer = LofLime(classifier, embeddings=x_train)
        attr["lof_lime"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "lof_kernel_shap" in explainers:
        explainer = LofKernelShap(classifier, embeddings=x_train)
        attr["lof_kernel_shap"] = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            feature_mask=seg_test,
            show_progress=True,
        )

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(classifier, data=x_train)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    if "occlusion" in explainers:
        explainer = Occlusion(classifier)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            baselines=baselines,
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    with open("results.csv", "a") as fp:
        for topk in areas:
            for k, v in attr.items():

                acc = accuracy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                )
                comp = comprehensiveness(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                )
                ce = cross_entropy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                )
                l_odds = log_odds(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                )
                suff = sufficiency(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
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
        "--n-segments",
        type=int,
        default=20,
        help="Number of segmentations.",
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
        n_segments=args.n_segments,
        accelerator=args.accelerator,
        seed=args.seed,
        deterministic=args.deterministic,
    )
