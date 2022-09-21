import multiprocessing as mp
import numpy as np
import os
import torch as th
import torchvision.transforms as T
import warnings

from captum.attr import (
    DeepLift,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    Lime,
)
from captum.metrics import sensitivity_max
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from skimage.segmentation import slic
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from typing import Callable, List, Tuple, Union

from tint.attr import (
    AugmentedOcclusion,
    BayesLime,
    BayesKernelShap,
    BayesMask,
    GeodesicIntegratedGradients,
    LofLime,
    LofKernelShap,
    Occlusion,
)
from tint.attr.models import BayesMaskNet
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


from experiments.mnist.classifier import MnistClassifierNet
from utils import experiment_dict


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def compute_attr(
    inputs: TensorOrTupleOfTensorsGeneric,
    explainer,
    baselines: BaselineType,
    target: TargetType,
    additional_forward_args: Union[None, Tensor, Tuple[Tensor, ...]],
):
    if isinstance(explainer, Occlusion):
        attr = explainer.attribute(
            inputs,
            sliding_window_shapes=(1, 1, 1),
            target=target,
            attributions_fn=abs,
        )
    elif isinstance(explainer, Lime):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
            feature_mask=additional_forward_args,
        )
    elif isinstance(explainer, IntegratedGradients):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
            internal_batch_size=200,
        )
    elif isinstance(explainer, DeepLift):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
        )
    elif isinstance(explainer, InputXGradient):
        attr = explainer.attribute(inputs, target=target)
    else:
        raise NotImplementedError

    return attr


def compute_metric(
    inputs: TensorOrTupleOfTensorsGeneric,
    explainer,
    metric: Callable,
    forward_func: Callable,
    baselines: BaselineType,
    topk: float,
    target: TargetType,
    additional_forward_args: Union[None, Tensor, Tuple[Tensor, ...]],
    weight_fn: Callable,
):
    attr = compute_attr(
        inputs=inputs,
        explainer=explainer,
        baselines=baselines,
        target=target,
        additional_forward_args=additional_forward_args,
    )

    metric_ = metric(
        forward_func,
        inputs,
        attributions=attr,
        baselines=baselines,
        topk=topk,
        weight_fn=weight_fn,
    )

    return tuple(th.ones_like(i) * metric_ for i in inputs)


def main(
    explainers: List[str],
    experiment: str,
    areas: List[float],
    n_segments: int = 20,
    accelerator: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
):
    # If experiment is provided, get list of explainers
    if experiment is not None:
        explainers = experiment_dict[experiment]

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
        max_epochs=10,
        accelerator=accelerator,
        devices=1,
        deterministic=deterministic,
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
        seg_test.append(th.from_numpy(segments) - 1)

    x_test = th.stack(x_test).float().to(accelerator)
    y_test = th.stack(y_test).long().to(accelerator)
    seg_test = th.stack(seg_test).to(accelerator)

    # Get train data for LOF and AugmentedOcclusion methods
    x_train = list()
    for data, _ in mnist_train_loader:
        x_train.append(data)
    x_train = th.cat(x_train).to(accelerator)

    # Create dict of attributions and explainers
    attr = dict()
    expl = dict()

    # Baselines is the normalised background
    baselines = -0.4242

    if "bayes_lime" in explainers:
        explainer = BayesLime(classifier)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["bayes_lime"] = th.stack(_attr)
        expl["bayes_lime"] = explainer

    if "bayes_kernel_shap" in explainers:
        explainer = BayesKernelShap(classifier)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["bayes_kernel_shap"] = th.stack(_attr)
        expl["bayes_kernel_shap"] = explainer

    if "bayes_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=1,
            log_every_n_steps=2,
            deterministic=deterministic,
        )
        mask = BayesMaskNet(
            forward_func=classifier,
            distribution="normal",
            hard=False,
            eps=1e-5,
            optim="adam",
            lr=0.01,
        )
        explainer = BayesMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
        )
        attr["bayes_mask"] = _attr.to(accelerator)
        expl["bayes_mask"] = explainer

    if "deep_lift" in explainers:
        explainer = DeepLift(classifier)
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
        )
        attr["deep_lift"] = _attr
        expl["deep_lift"] = explainer

    if "geodesic_integrated_gradients" in explainers:
        explainer = GeodesicIntegratedGradients(
            classifier,
            data=x_train,
            n_neighbors=5,
        )
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            internal_batch_size=200,
        )
        attr["geodesic_integrated_gradients"] = _attr
        expl["geodesic_integrated_gradients"] = explainer

    if "input_x_gradient" in explainers:
        explainer = InputXGradient(classifier)
        _attr = explainer.attribute(x_test, target=y_test)
        attr["input_x_gradient"] = _attr
        expl["input_x_gradient"] = explainer

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(classifier)
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            internal_batch_size=200,
        )
        attr["integrated_gradients"] = _attr
        expl["integrated_gradients"] = explainer

    if "lime" in explainers:
        explainer = Lime(classifier)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lime"] = th.stack(_attr)
        expl["lime"] = explainer

    if "kernel_shap" in explainers:
        explainer = KernelShap(classifier)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["kernel_shap"] = th.stack(_attr)
        expl["kernel_shap"] = explainer

    if "lof_lime" in explainers:
        explainer = LofLime(classifier, embeddings=x_train)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lof_lime"] = th.stack(_attr)
        expl["lof_lime"] = explainer

    if "lof_kernel_shap" in explainers:
        explainer = LofKernelShap(classifier, embeddings=x_train)
        _attr = list()
        for x, y, s in get_progress_bars()(
            zip(x_test, y_test, seg_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.unsqueeze(0),
                    feature_mask=s.unsqueeze(0),
                )
            )
        attr["lof_kernel_shap"] = th.stack(_attr)
        expl["lof_kernel_shap"] = explainer

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(classifier, data=x_train)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1, 1, 1),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )
        expl["augmented_occlusion"] = explainer

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
        expl["occlusion"] = explainer

    lime_weights_fn = lime_weights(
        distance_mode="euclidean", kernel_width=1000
    )
    lof_weights_fn = lof_weights(data=x_train, n_neighbors=20)

    with open("results.csv", "a") as fp, mp.Lock():
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
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    comp = comprehensiveness(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    ce = cross_entropy(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    l_odds = log_odds(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    suff = sufficiency(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        weight_fn=weight_fn,
                    )
                    if k != "bayes_mask":
                        sens_max = sensitivity_max(
                            compute_attr,
                            x_test[:100],
                            explainer=expl[k],
                            baselines=baselines,
                            target=y_test[:100],
                            additional_forward_args=seg_test[:100],
                        )
                        lip_max = lipschitz_max(
                            compute_attr,
                            x_test[:100],
                            explainer=expl[k],
                            baselines=baselines,
                            target=y_test[:100],
                            additional_forward_args=seg_test[:100],
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
                    if k != "bayes_mask":
                        fp.write(f"{sens_max.mean().item():.4},")
                        fp.write(f"{lip_max.mean().item():.4},")

                    if experiment == "lof":
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
                                forward_func=classifier,
                                baselines=baselines,
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
                                forward_func=classifier,
                                baselines=baselines,
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
        "--experiment",
        type=str,
        default=None,
        help="Which experiment to run. Ignored if None",
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
        experiment=args.experiment,
        areas=args.areas,
        n_segments=args.n_segments,
        accelerator=args.accelerator,
        seed=args.seed,
        deterministic=args.deterministic,
    )
