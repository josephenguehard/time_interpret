import multiprocessing as mp
import statistics
import random
import torch as th

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    TemporalAugmentedOcclusion,
    TemporalIntegratedGradients,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.datasets import Mimic3
from tint.metrics import mae, mse


from experiments.mimic3.blood_pressure.regressor import MimicRegressorNet


def main(
    explainers: List[str],
    areas: list,
    device: str = "cpu",
    fold: int = 0,
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

    # Load data
    mimic3 = Mimic3(task="blood_pressure", n_folds=5, fold=fold, seed=seed)

    # Create classifier
    regressor = MimicRegressorNet(
        feature_size=28,
        n_state=1,
        hidden_size=200,
        dropout=0.1,
        regres=True,
        loss="mse",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    trainer.fit(regressor, datamodule=mimic3)

    # Get data for explainers
    with lock:
        x_train = mimic3.preprocess(split="train")["x"].to(device)
        x_test = mimic3.preprocess(split="test")["x"].to(device)

    # Switch to eval
    regressor.eval()

    # Set model to device
    regressor.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(regressor))
        attr["deep_lift"] = (
            explainer.attribute(
                x_test,
                baselines=x_test * 0,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(regressor.cpu()))
        attr["gradient_shap"] = (
            explainer.attribute(
                x_test.cpu(),
                baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
                n_samples=50,
                stdevs=0.0001,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )
        regressor.to(device)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(regressor))
        attr["integrated_gradients"] = (
            explainer.attribute(
                x_test,
                baselines=x_test * 0,
                internal_batch_size=200,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                regressor, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = (
            explainer.attribute(
                x_test,
                sliding_window_shapes=(1,),
                attributions_fn=abs,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(regressor))
        attr["occlusion"] = (
            explainer.attribute(
                x_test,
                sliding_window_shapes=(1,),
                baselines=x_train.mean(0, keepdim=True),
                attributions_fn=abs,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )

    if "temporal_integrated_gradients" in explainers:
        explainer = TemporalIntegratedGradients(regressor)
        attr["temporal_integrated_gradients"] = (
            explainer.attribute(
                x_test,
                baselines=x_test * 0,
                internal_batch_size=200,
                return_temporal_attributions=True,
                show_progress=True,
            )
            .abs()
            .cpu()
        )

    # Regressor and x_test to cpu
    regressor.to("cpu")
    x_test = x_test.to("cpu")

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    # Compute metrics for each topk in areas.
    # We also get metrics for data up to each tim and average.
    with open("results.csv", "a") as fp, lock:
        for topk in areas:
            for k, v in attr.items():

                mae_comp = list()
                mae_suff = list()
                mse_comp = list()
                mse_suff = list()

                for time in range(x_test.shape[1]):
                    partial_x = x_test[:, : time + 1]
                    partial_x_avg = x_avg[:, : time + 1]
                    partial_attr = v[:, time, : time + 1]

                    mae_comp.append(
                        mae(
                            regressor,
                            partial_x,
                            attributions=partial_attr,
                            baselines=partial_x_avg,
                            topk=topk,
                            mask_largest=True,
                        )
                    )
                    mae_suff.append(
                        mae(
                            regressor,
                            partial_x,
                            attributions=partial_attr,
                            baselines=partial_x_avg,
                            topk=topk,
                            mask_largest=False,
                        )
                    )
                    mse_comp.append(
                        mse(
                            regressor,
                            partial_x,
                            attributions=partial_attr,
                            baselines=partial_x_avg,
                            topk=topk,
                            mask_largest=True,
                        )
                    )
                    mse_suff.append(
                        mse(
                            regressor,
                            partial_x,
                            attributions=partial_attr,
                            baselines=partial_x_avg,
                            topk=topk,
                            mask_largest=False,
                        )
                    )

                mae_comp = statistics.mean(mae_comp)
                mae_suff = statistics.mean(mae_suff)
                mse_comp = statistics.mean(mse_comp)
                mse_suff = statistics.mean(mse_suff)

                fp.write(str(seed) + ",")
                fp.write(str(fold) + ",")
                fp.write(str(topk) + ",")
                fp.write(k + ",")
                fp.write(f"{mae_comp:.4},")
                fp.write(f"{mae_suff:.4},")
                fp.write(f"{mse_comp:.4},")
                fp.write(f"{mse_suff:.4}")
                fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "deep_lift",
            "gradient_shap",
            "integrated_gradients",
            "augmented_occlusion",
            "occlusion",
            "temporal_integrated_gradients",
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
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
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
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
    )
