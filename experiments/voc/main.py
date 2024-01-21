import multiprocessing as mp
import os

import torch
import torch as th
import torchvision.transforms as T
import warnings

from captum.attr import (
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
)

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from typing import List

from tint.attr import (
    AugmentedOcclusion,
    DynaMask,
    ExtremalMask,
    GeodesicIntegratedGradients,
    GuidedIntegratedGradients,
    Occlusion,
)
from tint.attr.models import ExtremalMaskNet, MaskNet
from tint.metrics.white_box import (
    aup,
    aur,
    roc_auc,
    auprc,
    information,
    entropy,
)
from tint.models import CNN
from tint.utils import get_progress_bars

from experiments.voc.classifier import VocClassifierNet


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def main(
    explainers: List[str],
    model_name: str,
    n_images: int,
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

    # Create lock
    lock = mp.Lock()

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

    # Load train and val data
    voc_train = VOCSegmentation(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "voc",
        ),
        image_set="train",
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    voc_train_loader = DataLoader(voc_train, batch_size=32, shuffle=True)

    voc_val = VOCSegmentation(
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
    voc_val_loader = DataLoader(voc_val, batch_size=32, shuffle=False)

    # Load model
    model = VocClassifierNet(model=model_name)

    # Train classifier
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
    )
    trainer.fit(
        model,
        train_dataloaders=voc_train_loader,
        val_dataloaders=voc_val_loader,
    )

    # Extract net from classifier
    # Otherwise it fails when using deepcopy
    model = model.net

    # Switch to eval
    model.eval()

    # Set model to device
    model.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Get data as tensors
    # we only load n images
    voc_val_loader = DataLoader(voc_val, batch_size=1, shuffle=True)
    x_test = list()
    y_test = list()
    seg_test = list()
    i = 0
    for data, seg in voc_val_loader:
        if i == n_images:
            break

        # We remove images with multiple labels
        seg_ids = set(seg.unique().tolist()) - {0, 255}
        if len(seg_ids) != 1:
            continue

        # We set the segmentation data to 0-1
        (label,) = seg_ids
        seg = (seg == label).long().repeat(1, 3, 1, 1)

        # Get prediction from the model
        # We remove images which are not correctly predicted
        pred = model(data.to(device)).argmax(-1).cpu().item()
        if pred != label - 1:
            continue

        x_test.append(data)
        y_test.append(label - 1)
        seg_test.append(seg)
        i += 1

    x_test = th.cat(x_test).to(device)
    y_test = th.Tensor(y_test).long().to(device)
    seg_test = th.cat(seg_test).to(device)

    # Baseline is a normalised black image
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    baselines = normalizer(th.zeros_like(x_test))

    # Create dict of attributions
    attr = dict()

    # DeepLift not supported for ResNet
    if "deep_lift" in explainers:
        pass

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
        )
        mask = MaskNet(
            forward_func=model,
            perturbation="fade_moving_average",
            keep_ratio=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
        )
        explainer = DynaMask(model)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "extremal_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
        )
        mask = ExtremalMaskNet(
            forward_func=model,
            model=CNN([3, 3], kernel_size=3, padding=1, flatten=False),
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(model)
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=64,
        )
        attr["extremal_mask"] = _attr.to(device)

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()
        for i, (x, y, b) in get_progress_bars()(
            enumerate(zip(x_test, y_test, baselines)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            rand = th.rand((50,) + x.shape).sort(dim=0).values.to(device)
            x_aug = x.unsqueeze(0) * rand
            explainer = GeodesicIntegratedGradients(
                model, data=x_aug, n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=10,
                )
            )

        attr["geodesic_integrated_gradients"] = th.cat(_attr)

    if "enhanced_integrated_gradients" in explainers:
        _attr = list()
        for i, (x, y, b) in get_progress_bars()(
            enumerate(zip(x_test, y_test, baselines)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            rand = th.rand((50,) + x.shape).sort(dim=0).values.to(device)
            x_aug = x.unsqueeze(0) * rand
            explainer = GeodesicIntegratedGradients(
                model, data=x_aug, n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=10,
                    distance="euclidean",
                )
            )

        attr["enhanced_integrated_gradients"] = th.cat(_attr)

    if "guided_integrated_gradients" in explainers:
        _attr = list()
        explainer = GuidedIntegratedGradients(model)
        for i, (x, y, b) in get_progress_bars()(
            enumerate(zip(x_test, y_test, baselines)),
            total=len(x_test),
            desc=f"{GuidedIntegratedGradients.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=10,
                )
            )

        attr["guided_integrated_gradients"] = th.cat(_attr)

    if "gradient_shap" in explainers:
        explainer = GradientShap(model)
        _attr = list()
        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{GradientShap.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    th.stack([x, x * 0.0]),
                    target=y.item(),
                    n_samples=50,
                )
            )

        attr["gradient_shap"] = th.cat(_attr)

    if "noisy_gradient_shap" in explainers:
        explainer = GradientShap(model)
        _attr = list()
        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{GradientShap.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    normalizer(th.stack([x, x * 0.0])),
                    target=y.item(),
                    n_samples=50,
                    stdevs=1.0,
                )
            )

        attr["noisy_gradient_shap"] = th.cat(_attr)

    if "input_x_gradient" in explainers:
        explainer = InputXGradient(model)
        _attr = explainer.attribute(x_test, target=y_test)
        attr["input_x_gradient"] = _attr

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(model)
        _attr = explainer.attribute(
            x_test,
            baselines=baselines,
            target=y_test,
            internal_batch_size=200,
        )
        attr["integrated_gradients"] = _attr

    if "lime" in explainers:
        explainer = Lime(model)
        _attr = list()
        for x, y, b in get_progress_bars()(
            zip(x_test, y_test, baselines),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.unsqueeze(0),
                )
            )
        attr["lime"] = th.cat(_attr)

    if "kernel_shap" in explainers:
        explainer = KernelShap(model)
        _attr = list()
        for x, y, b in get_progress_bars()(
            zip(x_test, y_test, baselines),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.unsqueeze(0),
                )
            )
        attr["kernel_shap"] = th.cat(_attr)

    if "smooth_grad" in explainers:
        explainer = NoiseTunnel(IntegratedGradients(model))
        _attr = list()
        for i, (x, y, b) in get_progress_bars()(
            enumerate(zip(x_test, y_test, baselines)),
            total=len(x_test),
            desc=f"{NoiseTunnel.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    baselines=b.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=200,
                    nt_samples=10,
                    stdevs=1.0,
                    nt_type="smoothgrad_sq",
                )
            )

        attr["smooth_grad"] = th.cat(_attr)

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(model, data=x_test)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    if "occlusion" in explainers:
        explainer = Occlusion(model)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            baselines=baselines,
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    with open("results.csv", "a") as fp, lock:
        for k, v in get_progress_bars()(
            attr.items(), desc="Compute metrics", leave=False
        ):
            fp.write(str(seed) + ",")
            fp.write(model_name + ",")
            fp.write(k + ",")

            v = v.abs()
            for metric in [aup, aur, roc_auc, auprc, information, entropy]:
                result = list()

                for v_i, seg_i in zip(v, seg_test):
                    result.append(
                        metric(attributions=v_i, true_attributions=seg_i)
                    )

                result = th.Tensor(result).mean()
                fp.write(f"{result:.4}") if metric == entropy else fp.write(
                    f"{result:.4},"
                )

            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "dyna_mask",
            "extremal_mask",
            "geodesic_integrated_gradients",
            "enhanced_integrated_gradients",
            "guided_integrated_gradients",
            "gradient_shap",
            "noisy_gradient_shap",
            "input_x_gradient",
            "integrated_gradients",
            "lime",
            "kernel_shap",
            "augmented_occlusion",
            "occlusion",
            "smooth_grad",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet",
        choices=["resnet", "inception"],
        help="Model to explain.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=100,
        help="Number of images to use.",
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
        model_name=args.model_name,
        n_images=args.n_images,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
