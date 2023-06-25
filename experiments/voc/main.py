import multiprocessing as mp
import os
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
from tint.models import CNN, Net
from tint.utils import get_progress_bars

from experiments.voc.classifier import VocClassifier


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def main(
    explainers: List[str],
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
    resnet = Net(VocClassifier(), loss="cross_entropy")

    # Train classifier
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
    )
    trainer.fit(
        resnet,
        train_dataloaders=voc_train_loader,
        val_dataloaders=voc_val_loader,
    )

    # Extract net from classifier
    # Otherwise it fails when using deepcopy
    resnet = resnet.net

    # Switch to eval
    resnet.eval()

    # Set model to device
    resnet.to(device)

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
        pred = resnet(data).argmax(-1).item()
        if pred != label:
            continue

        x_test.append(data)
        y_test.append(label)
        seg_test.append(seg)
        i += 1

    x_test = th.cat(x_test).to(device)
    y_test = th.Tensor(y_test).long().to(device)
    seg_test = th.cat(seg_test).to(device)

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
            forward_func=resnet,
            perturbation="fade_moving_average",
            keep_ratio=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
        )
        explainer = DynaMask(resnet)
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
            forward_func=resnet,
            model=CNN([3, 3], kernel_size=3, padding=1, flatten=False),
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(resnet)
        _attr = explainer.attribute(
            x_test,
            target=y_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=64,
        )
        attr["extremal_mask"] = _attr.to(device)

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()
        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            rand = th.rand((50,) + x.shape).sort(dim=0).values.to(device)
            x_aug = x.unsqueeze(0) * rand
            explainer = GeodesicIntegratedGradients(
                resnet, data=x_aug, n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=10,
                )
            )

        attr["geodesic_integrated_gradients"] = th.cat(_attr)

    if "enhanced_integrated_gradients" in explainers:
        _attr = list()
        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            rand = th.rand((50,) + x.shape).sort(dim=0).values.to(device)
            x_aug = x.unsqueeze(0) * rand
            explainer = GeodesicIntegratedGradients(
                resnet, data=x_aug, n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=10,
                    distance="euclidean",
                )
            )

        attr["enhanced_integrated_gradients"] = th.cat(_attr)

    if "gradient_shap" in explainers:
        explainer = GradientShap(resnet)
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
        explainer = GradientShap(resnet)
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
                    stdevs=1.0,
                )
            )

        attr["noisy_gradient_shap"] = th.cat(_attr)

    if "input_x_gradient" in explainers:
        explainer = InputXGradient(resnet)
        _attr = explainer.attribute(x_test, target=y_test)
        attr["input_x_gradient"] = _attr

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(resnet)
        _attr = explainer.attribute(
            x_test,
            target=y_test,
            internal_batch_size=200,
        )
        attr["integrated_gradients"] = _attr

    if "lime" in explainers:
        explainer = Lime(resnet)
        _attr = list()
        for x, y in get_progress_bars()(
            zip(x_test, y_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                )
            )
        attr["lime"] = th.stack(_attr)

    if "kernel_shap" in explainers:
        explainer = KernelShap(resnet)
        _attr = list()
        for x, y in get_progress_bars()(
            zip(x_test, y_test),
            total=len(x_test),
            desc=f"{explainer.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.unsqueeze(0),
                )
            )
        attr["kernel_shap"] = th.stack(_attr)

    if "noise_tunnel" in explainers:
        explainer = NoiseTunnel(IntegratedGradients(resnet))
        _attr = list()
        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{NoiseTunnel.get_name()} attribution",
        ):
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=200,
                    nt_samples=10,
                    stdevs=1.0,
                    nt_type="smoothgrad_sq",
                )
            )

        attr["noise_tunnel"] = th.cat(_attr)

    if "augmented_occlusion" in explainers:
        explainer = AugmentedOcclusion(resnet, data=x_test)
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    if "occlusion" in explainers:
        explainer = Occlusion(resnet)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            target=y_test,
            attributions_fn=abs,
            show_progress=True,
        )

    with open("results.csv", "a") as fp, lock:
        for k, v in get_progress_bars()(
            attr.items(), desc="Compute metrics", leave=False
        ):
            v = v.abs()
            _aup = aup(attributions=v, true_attributions=seg_test)
            _aur = aur(attributions=v, true_attributions=seg_test)
            _roc_auc = roc_auc(attributions=v, true_attributions=seg_test)
            _auprc = auprc(attributions=v, true_attributions=seg_test)
            _info = information(attributions=v, true_attributions=seg_test)
            _entropy = entropy(attributions=v, true_attributions=seg_test)

            fp.write(str(seed) + ",")
            fp.write(k + ",")
            fp.write(f"{_aup:.4},")
            fp.write(f"{_aur:.4},")
            fp.write(f"{_roc_auc:.4},")
            fp.write(f"{_auprc:.4},")
            fp.write(f"{_info:.4},")
            fp.write(f"{_entropy:.4}")
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
            "gradient_shap",
            "noisy_gradient_shap",
            "input_x_gradient",
            "integrated_gradients",
            "lime",
            "kernel_shap",
            "noise_tunnel",
            "augmented_occlusion",
            "occlusion",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
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
        n_images=args.n_images,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
