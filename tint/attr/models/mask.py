import numpy as np
import torch as th
import torch.nn as nn

from captum._utils.common import _run_forward
from captum._utils.typing import TargetType

from typing import Callable, Union

from tint.models import Net


EPS = 1e-7
TIME_DIM = 1


class Mask(nn.Module):
    """
    Mask network for DynaMask method.

    Args:
        forward_func (Callable): The function to get prediction from.
        perturbation (str): Which perturbation to apply.
            Default to ``'fade_moving_average'``
        deletion_mode (bool): ``True`` if the mask should identify the most
            impactful deletions. Default to ``False``
        initial_mask_coef (float): Which value to use to initialise the mask.
            Default to 0.5
        keep_ratio (float): Fraction of elements in x that should be kept by
            the mask. Default to 0.5
        size_reg_factor_init (float): Initial coefficient for the regulator
            part of the total loss. Default to 0.5
        size_reg_factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100
        time_reg_factor (float): Regulation factor for the variation in time.
            Default to 0.0

    References:
        https://arxiv.org/pdf/2106.05303
    """

    def __init__(
        self,
        forward_func: Callable,
        perturbation: str = "fade_moving_average",
        deletion_mode: bool = False,
        initial_mask_coef: float = 0.5,
        keep_ratio: float = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100.0,
        time_reg_factor: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        assert perturbation in [
            "fade_moving_average",
            "gaussian_blur",
            "fade_moving_average_window",
            "fade_reference",
        ], f"{perturbation} perturbation not recognised."

        self.forward_func = forward_func
        self.perturbation = perturbation
        self.deletion_mode = deletion_mode
        self.initial_mask_coef = initial_mask_coef
        self.keep_ratio = keep_ratio
        self.size_reg_factor = size_reg_factor_init
        self.reg_multiplier = np.exp(np.log(size_reg_factor_dilation))
        self.time_reg_factor = time_reg_factor
        self.kwargs = kwargs

        self.register_parameter("mask", None)
        self.reg_ref = None

    def init(self, shape: tuple, n_epochs: int):
        # Create mask param
        self.mask = nn.Parameter(th.ones(*shape) * 0.5)

        # Init the regularisation parameter
        reg_ref = th.zeros_like(self.mask).reshape(len(self.mask), -1)
        reg_ref[
            :, int((1.0 - self.keep_ratio) * reg_ref.shape[TIME_DIM]) :
        ] = 1.0
        self.reg_ref = reg_ref

        # Update multiplier with n_epochs
        self.reg_multiplier /= n_epochs

    def fade_moving_average(self, x):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        moving_average = th.mean(x, TIME_DIM).unsqueeze(TIME_DIM)
        return mask * x + (1 - mask) * moving_average

    def gaussian_blur(self, x, sigma_max=2):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        t_axis = th.arange(1, x.shape[TIME_DIM] + 1).int()

        # Convert the mask into a tensor containing the width of each
        # Gaussian perturbation
        sigma_tensor = (sigma_max * ((1 + EPS) - mask)).unsqueeze(1)

        # For each feature and each time, we compute the coefficients for
        # the Gaussian perturbation
        t1_tensor = t_axis.unsqueeze(1).unsqueeze(2)
        t2_tensor = t_axis.unsqueeze(0).unsqueeze(2)
        filter_coefs = th.exp(
            th.divide(
                -1.0 * (t1_tensor - t2_tensor) ** 2, 2.0 * (sigma_tensor**2)
            )
        )
        filter_coefs = th.divide(filter_coefs, th.sum(filter_coefs, 0) + EPS)

        # The perturbation is obtained by replacing each input by the
        # linear combination weighted by Gaussian coefs
        return th.einsum("bsti,bsi->bti", filter_coefs, x)

    def fade_moving_average_window(self, x, window_size=2):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        t_axis = th.arange(1, x.shape[TIME_DIM] + 1).int()

        # For each feature and each time, we compute the coefficients
        # of the perturbation tensor
        t1_tensor = t_axis.unsqueeze(1)
        t2_tensor = t_axis.unsqueeze(0)
        filter_coefs = th.abs(t1_tensor - t2_tensor) <= window_size
        filter_coefs = filter_coefs / (2 * window_size + 1)
        x_avg = th.einsum("st,bsi->bti", filter_coefs, x)

        # The perturbation is just an affine combination of the input
        # and the previous tensor weighted by the mask
        return x_avg + mask * (x - x_avg)

    def fade_reference(self, x, x_ref):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        return x_ref + mask * (x - x_ref)

    def forward(self, x: th.Tensor, target: TargetType, *args) -> th.Tensor:
        # Clamp mask
        self.clamp()

        # Get perturbed input
        x_pert = getattr(self, self.perturbation)(x, **self.kwargs)

        # Return f(perturbed x)
        return _run_forward(
            forward_func=self.forward_func,
            inputs=x_pert,
            target=target,
            additional_forward_args=args,
        )

    def regularisation(self, loss: th.Tensor) -> th.Tensor:
        # Get size regularisation
        mask_sorted = self.mask.reshape(len(self.mask), -1).sort().values
        size_reg = ((self.reg_ref - mask_sorted) ** 2).mean()

        # Get time regularisation
        time_reg = (
            th.abs(
                self.mask[:, 1 : self.mask.shape[TIME_DIM] - 1, ...]
                - self.mask[:, : self.mask.shape[TIME_DIM] - 2, ...]
            )
        ).mean()

        # Return loss plus regularisation
        return (
            (1.0 - 2 * self.deletion_mode) * loss
            + self.size_reg_factor * size_reg
            + self.time_reg_factor * time_reg
        )

    def clamp(self):
        self.mask.data = self.mask.data.clamp(0, 1)

    def representation(self):
        return self.mask.detach().cpu()


class MaskNet(Net):
    """
    Mask network as a Pytorch Lightning module.

    Args:
        forward_func (Callable): The function to get prediction from.
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
        forward_func: Callable,
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
        **kwargs,
    ):
        mask = Mask(
            forward_func=forward_func,
            perturbation=perturbation,
            initial_mask_coef=initial_mask_coef,
            keep_ratio=keep_ratio,
            size_reg_factor_init=size_reg_factor_init,
            size_reg_factor_dilation=size_reg_factor_dilation,
            **kwargs,
        )

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, target, *args = batch
        y_hat = self(x.float(), target, *args)
        loss = self._loss(
            y_hat,
            _run_forward(
                forward_func=self.net.forward_func,
                inputs=y,
                target=target,
                additional_forward_args=tuple(args),
            )
        )
        return loss

    def training_step_end(self, step_output):
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        return step_output

    def training_epoch_end(self, outputs) -> None:
        # Increase the regulator coefficient
        self.net.size_reg_factor *= self.net.reg_multiplier
