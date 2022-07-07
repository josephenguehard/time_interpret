import numpy as np
import torch as th
import torch.nn as nn

from typing import Callable, Union

from tint.models import Net


class Mask(nn.Module):
    """
    Mask network for DynaMask method.

    Args:
        perturbation (str): Which perturbation to apply.
            Default to ``'fade_moving_average'``
        initial_mask_coef (float): Which value to use to initialise the mask.
            Default to 0.5
        keep_ratio (float): Fraction of elements in x that should be kept by
            the mask. Default to 0.5
        size_reg_factor_init (float): Initial coefficient for the regulator
            part of the total loss. Default to 0.5
        size_reg_factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100

    References:
        https://arxiv.org/pdf/2106.05303
    """

    def __init__(
        self,
        perturbation: str = "fade_moving_average",
        initial_mask_coef: float = 0.5,
        keep_ratio: float = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100.0,
    ):
        super().__init__()

        assert perturbation in [
            "fade_moving_average",
            "gaussian_blur",
            "fade_moving_average_window",
        ], "perturbation not recognised."

        self.perturbation = perturbation
        self.initial_mask_coef = initial_mask_coef
        self.keep_ratio = keep_ratio
        self.reg_factor = size_reg_factor_init
        self.reg_multiplier = np.exp(np.log(size_reg_factor_dilation))

        self.register_parameter("mask", None)
        self.reg_ref = None

    def init(self, shape: tuple, n_epochs: int):
        # Create mask param
        self.mask = nn.Parameter(th.Tensor(*shape))

        # Init the regularisation parameter
        reg_ref = th.zeros_like(self.mask).reshape(-1)
        reg_ref[int(self.keep_ratio * len(reg_ref)) :] = 1.0
        self.reg_ref = reg_ref

        # Update multiplier with n_epochs
        self.reg_multiplier /= n_epochs

    def fade_moving_average(self, x):
        moving_average = th.mean(x, 0).reshape(1, -1)
        moving_average_tiled = moving_average.repeat(len(x), 1)
        return self.mask * x + (1 - self.mask) * moving_average_tiled

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = getattr(self, self.perturbation)(x)
        return super().forward(x)

    def loss(self, loss: th.Tensor) -> th.Tensor:
        mask_sorted = self.mask.reshape(-1).sort()[0]
        size_reg = ((self.reg_ref - mask_sorted) ** 2).mean()
        return loss + size_reg

    def clamp(self):
        self.mask.data = self.mask.data.clamp(0, 1)

    def representation(self):
        return self.mask.detach()


class MaskNet(Net):
    """
    Mask network as a Pytorch Lightning module.

    Args:
        perturbation (str): Which perturbation to apply.
            Default to ``'fade_moving_average'``
        initial_mask_coef (float): Which value to use to initialise the mask.
            Default to 0.5
        keep_ratio (float): Fraction of elements in x that should be kept by
            the mask. Default to 0.5
        size_reg_factor_init (float): Initial coefficient for the regulator
            part of the total loss. Default to 0.5
        size_reg_factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0
    """

    def __init__(
        self,
        perturbation: str = "fade_moving_average",
        initial_mask_coef: float = 0.5,
        keep_ratio: float = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100.0,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = Mask(
            perturbation=perturbation,
            initial_mask_coef=initial_mask_coef,
            keep_ratio=keep_ratio,
            size_reg_factor_init=size_reg_factor_init,
            size_reg_factor_dilation=size_reg_factor_dilation,
        )

        super().__init__(
            layers=[mask],
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def training_step_end(self, step_output):
        self.net[0].clamp()
        return step_output

    def training_epoch_end(self, outputs) -> None:
        self.net[0].reg_factor *= self.net[0].reg_multiplier
