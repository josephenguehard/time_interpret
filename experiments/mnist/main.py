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
from typing import List, Tuple, Union

from tint.attr import (
    AugmentedOcclusion,
    BayesMask,
    DynaMask,
    GeodesicIntegratedGradients,
    Occlusion,
)
from tint.attr.models import BayesMaskNet, MaskNet
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    lipschitz_max,
    log_odds,
    sufficiency,
)
from tint.utils import get_progress_bars


from experiments.mnist.classifier import MnistClassifierNet


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def compute_attr(
    inputs: TensorOrTupleOfTensorsGeneric,
    explainer,
    baselines: BaselineType,
    target: TargetType,
    additional_forward_args: Union[None, Tensor, Tuple[Tensor, ...]],
):
    if isinstance(explainer, DeepLift):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
        )
    elif isinstance(explainer, GeodesicIntegratedGradients):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
            internal_batch_size=200,
        )
    elif isinstance(explainer, IntegratedGradients):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
            internal_batch_size=200,
        )
    elif isinstance(explainer, InputXGradient):
        attr = explainer.attribute(inputs, target=target)
    elif isinstance(explainer, Lime):
        attr = explainer.attribute(
            inputs,
            baselines=baselines,
            target=target,
            feature_mask=additional_forward_args,
        )
    elif isinstance(explainer, Occlusion):
        attr = explainer.attribute(
            inputs,
            sliding_window_shapes=(1, 1, 1),
            target=target,
            attributions_fn=abs,
        )
    else:
        raise NotImplementedError

    return attr


def main(
    explainers: List[str],
    areas: List[float],
    n_segments: int = 20,
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
        devices=device_id,
        deterministic=deterministic,
    )
    trainer.fit(
        classifier,
        train_dataloaders=mnist_train_loader,
        val_dataloaders=mnist_test_loader,
    )

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

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

    x_test = th.stack(x_test).float().to(device)
    y_test = th.stack(y_test).long().to(device)
    seg_test = th.stack(seg_test).to(device)

    # Get train data for LOF and AugmentedOcclusion methods
    x_train = list()
    for data, _ in mnist_train_loader:
        x_train.append(data)
    x_train = th.cat(x_train).to(device)

    # Create dict of attributions and explainers
    attr = dict()
    expl = dict()

    # Baselines is the normalised background
    baselines = -0.4242

    if "bayes_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
        )
        mask = BayesMaskNet(
            forward_func=classifier,
            distribution="none",
            hard=False,
            comp_loss=True,
            optim="adam",
            lr=0.01,
        )
        explainer = BayesMask(classifier)
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=64,
        )
        attr["bayes_mask"] = _attr.to(device)
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

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="fade_moving_average",
            keep_ratio=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()
        _sens_max = list()
        _lip_max = list()
        for i, (x, y, s) in get_progress_bars()(
            enumerate(zip(x_test, y_test, seg_test)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            rand = th.rand((500,) + x.shape).sort(dim=0).values.to(device)
            x_aug = (x - baselines).unsqueeze(0) * rand + baselines
            explainer = GeodesicIntegratedGradients(
                classifier, data=x_aug, n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=baselines,
                    target=y.item(),
                    internal_batch_size=200,
                )
            )

            if i < 100:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        baselines=baselines,
                        target=y.item(),
                        additional_forward_args=None,
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        baselines=baselines,
                        target=y.item(),
                        additional_forward_args=None,
                    )
                )

        attr["geodesic_integrated_gradients"] = th.cat(_attr)
        _sens_max = th.cat(_sens_max)
        _lip_max = th.cat(_lip_max)

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

    with open("results.csv", "a") as fp, mp.Lock():
        for k, v in get_progress_bars()(
            attr.items(), desc="Attr", leave=False
        ):
            if k not in ["bayes_mask", "dyna_mask"]:
                if k == "geodesic_integrated_gradients":
                    sens_max = _sens_max
                    lip_max = _lip_max
                else:
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
                fp.write("None,")
                fp.write(k + ",")
                fp.write("None,")
                fp.write("None,")
                fp.write("None,")
                fp.write("None,")
                fp.write("None,")
                fp.write("None,")
                fp.write("None,")
                fp.write(f"{sens_max.mean().item():.4},")
                fp.write(f"{lip_max.mean().item():.4}")
                fp.write("\n")

        for topk in get_progress_bars()(areas, desc="Topk", leave=False):
            for k, v in get_progress_bars()(
                attr.items(), desc="Attr", leave=False
            ):
                acc_comp = accuracy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                    mask_largest=True,
                )
                acc_suff = accuracy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                    mask_largest=False,
                )
                comp = comprehensiveness(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                )
                ce_comp = cross_entropy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                    mask_largest=True,
                )
                ce_suff = cross_entropy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                    mask_largest=False,
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
                fp.write(f"{acc_comp:.4},")
                fp.write(f"{acc_suff:.4},")
                fp.write(f"{comp:.4},")
                fp.write(f"{ce_comp:.4},")
                fp.write(f"{ce_suff:.4},")
                fp.write(f"{l_odds:.4},")
                fp.write(f"{suff:.4},")
                fp.write("None,")
                fp.write("None")
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
            "geodesic_integrated_gradients",
            "input_x_gradient",
            "integrated_gradients",
            "lime",
            "kernel_shap",
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
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
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
        areas=args.areas,
        n_segments=args.n_segments,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
