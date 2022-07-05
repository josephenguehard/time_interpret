import pytorch_lightning as pl
import torch as th

from .grad import grad


def smooth_grad(
    model: pl.LightningModule,
    x: th.Tensor,
    num_samples: int,
    noise_level: float,
) -> th.Tensor:
    """
    Guided Backpropagation method.

    Args:
        model (pl.LightningModule): The model to interpret.
        x (th.Tensor): Input data.
        num_samples (int): Number of noisy samples to create.
        noise_level (float): How much noise to add.

    Returns:
        th.Tensor: Gradients.

    References:
        https://arxiv.org/pdf/1706.03825
    """
    # Repeat x along batch dimension
    x_shape = x.shape
    x = x.repeat((num_samples,) + (1,) * (len(x_shape) - 1))

    # Add noise to data
    x = add_noise(x=x, noise_level=noise_level)

    # Compute gradients and return
    return grad(model=model, x=x) / num_samples


def add_noise(x: th.Tensor, noise_level: float):
    """
    Add gaussian noise to x

    Args:
        x (th.Tensor): Input data.
        noise_level (float): How much noise to add.
    """
    mean = 0.0
    std = noise_level * x.std().item()
    x = x.add_(th.zeros(x.size()).normal_(mean, std))
    return x.detach().requires_grad_(True)
