import multiprocessing as mp
import os
import torch as th
import torchvision.transforms as T
import warnings

from captum.attr import (
    DeepLift,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
)
from captum.metrics import sensitivity_max
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from typing import Any, List

from tint.attr import (
    AugmentedOcclusion,
    DynaMask,
    ExtremalMask,
    GeodesicIntegratedGradients,
    Occlusion,
)
from tint.attr.models import ExtremalMaskNet, MaskNet
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    lipschitz_max,
    log_odds,
    sufficiency,
)
from tint.models import CNN
from tint.utils import get_progress_bars

from experiments.voc.classifier import VocClassifier


file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


def compute_attr(
    inputs: TensorOrTupleOfTensorsGeneric,
    explainer,
    classifier,
    target: TargetType,
    additional_forward_args: Any,
):
    if isinstance(explainer, DeepLift):
        attr = explainer.attribute(
            inputs,
            target=target,
        )

    elif isinstance(explainer, GeodesicIntegratedGradients):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(additional_forward_args, tuple):
            additional_forward_args = additional_forward_args[0]

        _attr = list()
        for x in inputs:
            rand = th.rand((50,) + x.shape).sort(dim=0).values.to(x.device)
            x_aug = x.unsqueeze(0) * rand
            _explainer = GeodesicIntegratedGradients(
                classifier, data=x_aug, n_neighbors=5
            )

            _attr.append(
                _explainer.attribute(
                    x.unsqueeze(0),
                    target=target,
                    internal_batch_size=10,
                    distance=additional_forward_args,
                )
            )
        attr = th.cat(_attr)

    elif isinstance(explainer, GradientShap):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(additional_forward_args, tuple):
            additional_forward_args = additional_forward_args[0]

        _attr = list()
        for x in inputs:
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    th.stack([x, x * 0.0]),
                    target=target,
                    n_samples=50,
                    stdevs=additional_forward_args,
                )
            )
        attr = th.cat(_attr)

    elif isinstance(explainer, IntegratedGradients):
        attr = explainer.attribute(
            inputs,
            target=target,
            internal_batch_size=200,
        )

    elif isinstance(explainer, InputXGradient):
        attr = explainer.attribute(inputs, target=target)

    elif isinstance(explainer, Lime):
        attr = explainer.attribute(
            inputs,
            target=target,
            feature_mask=additional_forward_args,
        )

    elif isinstance(explainer, NoiseTunnel):
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        _attr = list()
        for x in inputs:
            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=target,
                    internal_batch_size=200,
                    nt_samples=10,
                    stdevs=1.0,
                    nt_type="smoothgrad_sq",
                )
            )
        attr = th.cat(_attr)

    elif isinstance(explainer, Occlusion):
        attr = explainer.attribute(
            inputs,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            target=target,
            attributions_fn=abs,
        )

    else:
        raise NotImplementedError

    if isinstance(attr, Tensor):
        attr = (attr,)
    return attr


def main(
    explainers: List[str],
    areas: List[float],
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
    resnet = VocClassifier()

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
    # we only load 100 images
    x_test = list()
    seg_test = list()
    i = 0
    for data, seg in voc_loader:
        if i == 100:
            break

        seg_ids = seg.unique()
        if len(seg_ids) <= 1:
            continue

        seg_ = seg.clone()
        for j, seg_id in enumerate(seg_ids):
            seg_[seg_ == seg_id] = j

        x_test.append(data)
        seg_test.append(seg_)
        i += 1

    x_test = th.cat(x_test).to(device)
    seg_test = th.cat(seg_test).to(device)

    # Target is the model prediction
    y_test = resnet(x_test).argmax(-1).to(device)

    # Create dict of attributions, explainers, sensitivity max
    # and lipschitz max
    attr = dict()
    expl = dict()
    sens_max_dict = dict()
    lip_max_dict = dict()

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
            deletion_loss=True,
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
        expl["extremal_mask"] = explainer

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()
        _sens_max = list()
        _lip_max = list()
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

            if i < 10:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args="geodesic",
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args="geodesic",
                    )
                )

        attr["geodesic_integrated_gradients"] = th.cat(_attr)
        sens_max_dict["geodesic_integrated_gradients"] = th.cat(_sens_max)
        lip_max_dict["geodesic_integrated_gradients"] = th.cat(_lip_max)

    if "enhanced_integrated_gradients" in explainers:
        _attr = list()
        _sens_max = list()
        _lip_max = list()
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

            if i < 10:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args="euclidean",
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args="euclidean",
                    )
                )

        attr["enhanced_integrated_gradients"] = th.cat(_attr)
        sens_max_dict["enhanced_integrated_gradients"] = th.cat(_sens_max)
        lip_max_dict["enhanced_integrated_gradients"] = th.cat(_lip_max)

    if "gradient_shap" in explainers:
        explainer = GradientShap(resnet)
        _attr = list()
        _sens_max = list()
        _lip_max = list()
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

            if i < 10:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=0.0,
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=0.0,
                    )
                )

        attr["gradient_shap"] = th.cat(_attr)
        sens_max_dict["gradient_shap"] = th.cat(_sens_max)
        lip_max_dict["gradient_shap"] = th.cat(_lip_max)

    if "noisy_gradient_shap" in explainers:
        explainer = GradientShap(resnet)
        _attr = list()
        _sens_max = list()
        _lip_max = list()
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

            if i < 10:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=1.0,
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=1.0,
                    )
                )

        attr["noisy_gradient_shap"] = th.cat(_attr)
        sens_max_dict["noisy_gradient_shap"] = th.cat(_sens_max)
        lip_max_dict["noisy_gradient_shap"] = th.cat(_lip_max)

    if "input_x_gradient" in explainers:
        explainer = InputXGradient(resnet)
        _attr = explainer.attribute(x_test, target=y_test)
        attr["input_x_gradient"] = _attr
        expl["input_x_gradient"] = explainer

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(resnet)
        _attr = explainer.attribute(
            x_test,
            target=y_test,
            internal_batch_size=200,
        )
        attr["integrated_gradients"] = _attr
        expl["integrated_gradients"] = explainer

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

    if "noise_tunnel" in explainers:
        explainer = NoiseTunnel(IntegratedGradients(resnet))
        _attr = list()
        _sens_max = list()
        _lip_max = list()
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

            if i < 10:
                _sens_max.append(
                    sensitivity_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=None,
                    )
                )
                _lip_max.append(
                    lipschitz_max(
                        compute_attr,
                        x.unsqueeze(0),
                        explainer=explainer,
                        classifier=resnet,
                        target=y.item(),
                        additional_forward_args=None,
                    )
                )

        attr["noise_tunnel"] = th.cat(_attr)
        sens_max_dict["noise_tunnel"] = th.cat(_sens_max)
        lip_max_dict["noise_tunnel"] = th.cat(_lip_max)

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
        expl["augmented_occlusion"] = explainer

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
        expl["occlusion"] = explainer

    with open("results.csv", "a") as fp, lock:
        for k, v in get_progress_bars()(
            attr.items(), desc="Attr", leave=False
        ):
            if k not in ["dyna_mask", "extremal_mask"]:
                if k in [
                    "geodesic_integrated_gradients",
                    "enhanced_integrated_gradients",
                    "gradient_shap",
                    "noisy_gradient_shap",
                    "noise_tunnel",
                ]:
                    sens_max = sens_max_dict[k]
                    lip_max = lip_max_dict[k]
                else:
                    sens_max = sensitivity_max(
                        compute_attr,
                        x_test[:10],
                        explainer=expl[k],
                        classifier=resnet,
                        target=y_test[:10],
                        additional_forward_args=seg_test[:10],
                    )
                    lip_max = lipschitz_max(
                        compute_attr,
                        x_test[:10],
                        explainer=expl[k],
                        classifier=resnet,
                        target=y_test[:10],
                        additional_forward_args=seg_test[:10],
                    )

                fp.write(str(seed) + ",")
                fp.write("None,")
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

        for mode in get_progress_bars()(
            ["zeros", "aug"], total=2, desc="Mode", leave=False
        ):
            for topk in get_progress_bars()(areas, desc="Topk", leave=False):
                for k, v in get_progress_bars()(
                    attr.items(), desc="Attr", leave=False
                ):
                    acc_comp = accuracy(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                        mask_largest=True,
                    )
                    acc_suff = accuracy(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                        mask_largest=False,
                    )
                    comp = comprehensiveness(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                    )
                    ce_comp = cross_entropy(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                        mask_largest=True,
                    )
                    ce_suff = cross_entropy(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                        mask_largest=False,
                    )
                    l_odds = log_odds(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                    )
                    suff = sufficiency(
                        resnet,
                        x_test,
                        attributions=v.cpu().abs(),
                        baselines=x_test if mode == "aug" else None,
                        n_samples=10 if mode == "aug" else 1,
                        n_samples_batch_size=1 if mode == "aug" else None,
                        stdevs=0.1 if mode == "aug" else 0.0,
                        draw_baseline_from_distrib=(
                            True if mode == "aug" else False
                        ),
                        topk=topk,
                    )

                    fp.write(str(seed) + ",")
                    fp.write(mode + ",")
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
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
