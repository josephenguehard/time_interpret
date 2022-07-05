import pytorch_lightning as pl
import torch as th

from .grad import grad


def grad_times_inputs(model: pl.LightningModule, x: th.Tensor) -> th.Tensor:
    """
    Input * Gradient method.

    Args:
        model (pl.LightningModule): The model to interpret.
        x (th.Tensor): Input data.

    Returns:
        th.Tensor: Gradients.

    References:
        https://arxiv.org/abs/1605.01713
    """
    return x * grad(model=model, x=x)
