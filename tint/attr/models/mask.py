import numpy as np
import torch as th
import torch.nn as nn

from captum._utils.common import (
    _expand_additional_forward_args,
    _run_forward,
)

from typing import Callable, List, Union

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
        keep_ratio (float, list): Fraction of elements in x that should be kept by
            the mask. Default to 0.5
        size_reg_factor_init (float): Initial coefficient for the regulator
            part of the total loss. Default to 0.5
        size_reg_factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100
        time_reg_factor (float): Regulation factor for the variation in time.
            Default to 0.0

    References:
        `Explaining Time Series Predictions with Dynamic Masks <https://arxiv.org/abs/2106.05303>`_
    """

    def __init__(
        self,
        forward_func: Callable,
        perturbation: str = "fade_moving_average",
        batch_size: int = 32,
        deletion_mode: bool = False,
        initial_mask_coef: float = 0.5,
        keep_ratio: Union[float, List[float]] = 0.5,
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

        object.__setattr__(self, "forward_func", forward_func)
        self.perturbation = perturbation
        self.batch_size = batch_size
        self.deletion_mode = deletion_mode
        self.initial_mask_coef = initial_mask_coef
        self.keep_ratio = (
            [keep_ratio] if isinstance(keep_ratio, float) else keep_ratio
        )
        self.size_reg_factor = size_reg_factor_init
        self.size_reg_factor_dilation = size_reg_factor_dilation
        self.time_reg_factor = time_reg_factor
        self.kwargs = kwargs

        self.register_parameter("mask", None)
        self.reg_ref = None
        self.reg_multiplier = None

    def init(self, shape: tuple, n_epochs: int, batch_size: int):
        # Create mask param
        shape = (len(self.keep_ratio) * shape[0],) + shape[1:]
        self.mask = nn.Parameter(th.ones(*shape) * 0.5)

        # Init the regularisation parameter
        reg_ref = th.zeros_like(self.mask).reshape(len(self.mask), -1)
        length = shape[0] // len(self.keep_ratio)
        for i, ratio in enumerate(self.keep_ratio):
            reg_ref[
                i * length : (i + 1) * length,
                int((1.0 - ratio) * reg_ref.shape[TIME_DIM]) :,
            ] = 1.0
        self.reg_ref = reg_ref

        # Update multiplier with n_epochs
        self.reg_multiplier = np.exp(
            np.log(self.size_reg_factor_dilation) / n_epochs
        )

        # Update batch size
        self.batch_size = batch_size

    def fade_moving_average(self, x, batch_idx):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        mask = mask[
            len(self.keep_ratio)
            * self.batch_size
            * batch_idx : len(self.keep_ratio)
            * self.batch_size
            * (batch_idx + 1)
        ]
        x = x.repeat((len(self.keep_ratio),) + (1,) * (len(x.shape) - 1))

        moving_average = th.mean(x, TIME_DIM).unsqueeze(TIME_DIM)
        return mask * x + (1 - mask) * moving_average

    def gaussian_blur(self, x, batch_idx, sigma_max=2):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        mask = mask[
            len(self.keep_ratio)
            * self.batch_size
            * batch_idx : len(self.keep_ratio)
            * self.batch_size
            * (batch_idx + 1)
        ]
        x = x.repeat((len(self.keep_ratio),) + (1,) * (len(x.shape) - 1))

        t_axis = th.arange(1, x.shape[TIME_DIM] + 1).int().to(x.device)

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
        filter_coefs = th.divide(
            filter_coefs, filter_coefs.sum(1, keepdim=True) + EPS
        )

        # The perturbation is obtained by replacing each input by the
        # linear combination weighted by Gaussian coefs
        return th.einsum("bsti,bsi->bti", filter_coefs, x)

    def fade_moving_average_window(self, x, batch_idx, window_size=2):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        mask = mask[
            len(self.keep_ratio)
            * self.batch_size
            * batch_idx : len(self.keep_ratio)
            * self.batch_size
            * (batch_idx + 1)
        ]
        x = x.repeat((len(self.keep_ratio),) + (1,) * (len(x.shape) - 1))

        t_axis = th.arange(1, x.shape[TIME_DIM] + 1).int().to(x.device)

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

    def fade_reference(self, x, batch_idx, x_ref):
        mask = 1.0 - self.mask if self.deletion_mode else self.mask
        mask = mask[
            len(self.keep_ratio)
            * self.batch_size
            * batch_idx : len(self.keep_ratio)
            * self.batch_size
            * (batch_idx + 1)
        ]
        x = x.repeat((len(self.keep_ratio),) + (1,) * (len(x.shape) - 1))

        return x_ref + mask * (x - x_ref)

    def forward(
        self, x: th.Tensor, batch_idx, *additional_forward_args
    ) -> th.Tensor:
        # Clamp mask
        self.clamp()

        # Get perturbed input
        x_pert = getattr(self, self.perturbation)(x, batch_idx, **self.kwargs)

        # Expand target and additional inputs when using several keep_ratio
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args, len(self.keep_ratio)
            )
            if additional_forward_args is not None
            else None
        )

        # Return f(perturbed x)
        return _run_forward(
            forward_func=self.forward_func,
            inputs=x_pert,
            additional_forward_args=input_additional_args,
        )

    def regularisation(self, loss: th.Tensor) -> th.Tensor:
        # Get size regularisation
        mask_sorted = self.mask.reshape(len(self.mask), -1).sort().values
        size_reg = (
            (self.reg_ref.to(self.mask.device) - mask_sorted) ** 2
        ).mean()

        # Get time regularisation if factor is positive
        time_reg = 0.0
        if self.time_reg_factor > 0:
            mask = self.mask.reshape(
                (len(self.mask) // len(self.keep_ratio), len(self.keep_ratio))
                + self.mask.shape[1:]
            )
            time_reg = (
                th.abs(
                    mask[:, :, 1 : self.mask.shape[TIME_DIM] - 1, ...]
                    - mask[:, :, : self.mask.shape[TIME_DIM] - 2, ...]
                )
            ).mean()

        # Return loss plus regularisation
        return (
            (1.0 - 2 * self.deletion_mode) * loss
            + self.size_reg_factor * size_reg
            + self.time_reg_factor * time_reg
        )

    def clamp(self):
        self.mask.data.clamp_(0, 1)


class MaskNet(Net):
    """
    Mask network as a Pytorch Lightning module.

    Args:
        forward_func (Callable): The function to get prediction from.
        perturbation (str): Which perturbation to apply.
            Default to ``'fade_moving_average'``
        deletion_mode (bool): ``True`` if the mask should identify the most
            impactful deletions. Default to ``False``
        initial_mask_coef (float): Which value to use to initialise the mask.
            Default to 0.5
        keep_ratio (float, list): Fraction of elements in x that should be kept by
            the mask. Default to 0.5
        size_reg_factor_init (float): Initial coefficient for the regulator
            part of the total loss. Default to 0.5
        size_reg_factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100
        time_reg_factor (float): Regulation factor for the variation in time.
            Default to 0.0
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    References:
        `Explaining Time Series Predictions with Dynamic Masks <https://arxiv.org/abs/2106.05303>`_

    Examples:
        >>> import numpy as np
        >>> import torch as th
        >>> from tint.attr.models import MaskNet
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> mask = MaskNet(
        ...     forward_func=mlp,
        ...     perturbation="gaussian_blur",
        ...     sigma_max=1,
        ...     keep_ratio=list(np.arange(0.25, 0.35, 0.01)),
        ...     size_reg_factor_init=0.1,
        ...     size_reg_factor_dilation=100,
        ...     time_reg_factor=1.0,
        ... )
    """

    def __init__(
        self,
        forward_func: Callable,
        perturbation: str = "fade_moving_average",
        batch_size: int = 32,
        deletion_mode: bool = False,
        initial_mask_coef: float = 0.5,
        keep_ratio: Union[float, List[float]] = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100.0,
        time_reg_factor: float = 0.0,
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
            batch_size=batch_size,
            deletion_mode=deletion_mode,
            initial_mask_coef=initial_mask_coef,
            keep_ratio=keep_ratio,
            size_reg_factor_init=size_reg_factor_init,
            size_reg_factor_dilation=size_reg_factor_dilation,
            time_reg_factor=time_reg_factor,
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

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get perturbed output
        if additional_forward_args is None:
            y_hat = self(x.float(), batch_idx)
        else:
            y_hat = self(x.float(), batch_idx, *additional_forward_args)

        # Get unperturbed output
        y_target = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
        y_target = th.cat([y_target] * len(self.net.keep_ratio), dim=0)

        # Uncomment these 2 lines to get the original state experiment.
        # The task is to predict the next time given the current
        # perturbed one. To be used with a cross-entropy loss. Please refer to
        # https://github.com/JonathanCrabbe/Dynamask/issues/4
        # for more details.
        # y_hat = y_hat.transpose(1, 2).reshape(y_hat.shape)
        # y_target = y_target.transpose(1, 2).reshape(y_target.shape)

        loss = self.loss(y_hat, y_target)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get unperturbed output
        y_target = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
        y_target = th.cat([y_target] * len(self.net.keep_ratio), dim=0)

        if additional_forward_args is None:
            return self(x.float(), batch_idx), y_target
        else:
            return (
                self(x.float(), batch_idx, *additional_forward_args),
                y_target,
            )

    def training_step_end(self, step_output):
        """"""
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        return step_output

    def on_train_epoch_end(self) -> None:
        # Increase the regulator coefficient
        self.net.size_reg_factor *= self.net.reg_multiplier

    def configure_optimizers(self):
        if self._optim == "adam":
            optim = th.optim.Adam(
                [
                    {"params": self.net.mask},
                ],
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                [
                    {"params": self.net.mask},
                ],
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}
