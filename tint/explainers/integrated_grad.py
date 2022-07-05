import pytorch_lightning as pl
import torch as th

from .grad import grad
from .smooth_grad import add_noise


def integrated_gradients(
    model: pl.LightningModule,
    x: th.Tensor,
    steps: int,
    baseline: th.Tensor = None,
    smooth: bool = False,
    noise_level: float = 0.2,
) -> th.Tensor:
    """
    Integrated Gradients method.

    Args:
        model (pl.LightningModule): The model to interpret.
        x (th.Tensor): Input data.
        steps (int): Number of steps to approximate integral.
        baseline (th.Tensor): Baseline data. If ``None``, create a tensor
            of zeros. Default to ``None``
        smooth (bool): Switch for combining with SmoothGrad.
            Default to ``False``
        noise_level (float): Defines the noise level of the additional
            Gaussian noise. Default to 0.2

    Returns:
        th.Tensor: Gradients.

    References:
        https://arxiv.org/pdf/1703.01365.pdf
    """
    # If no baseline, create one
    if baseline is None:
        baseline = th.zeros_like(x)

    # Compute difference between x and baseline
    diff = x - baseline

    # Create new inputs
    # If smooth, add Gaussian noise
    new_x = th.zeros((steps,) + x.shape)
    for step in range(steps):
        new_x[step] = (
            (baseline + (diff * step / steps)).detach().requires_grad_(True)
        )
        if smooth:
            new_x[step] = add_noise(x=new_x[step], noise_level=noise_level)

    # return gradients
    return grad(model=model, x=new_x).sum(0) / steps
