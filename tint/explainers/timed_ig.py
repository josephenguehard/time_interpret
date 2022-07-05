import pytorch_lightning as pl
import torch as th

from .grad import grad


def timed_integrated_gradients(
    model: pl.LightningModule,
    x: th.Tensor,
    steps: int,
    baseline: th.Tensor = None,
) -> th.Tensor:
    """
    Timed Integrated Gradients method.

    Args:
        model (pl.LightningModule): The model to interpret.
        x (th.Tensor): Input data.
        steps (int): Number of steps to approximate integral.
        baseline (th.Tensor): Baseline data. If ``None``, create a tensor
            of zeros. Default to ``None``

    Returns:
        th.Tensor: Gradients.
    """
    # If no baseline, create one
    if baseline is None:
        baseline = th.zeros_like(x)

    # Create new inputs
    # If smooth, add Gaussian noise
    new_x = th.zeros((steps,) + x.shape)
    for step in range(x.shape[1]):
        new_x[step] = baseline
        new_x[step][:, :step, ...] = x[:, :step, ...]

    # return gradients
    return grad(model=model, x=new_x).sum(0) / steps
