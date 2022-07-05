import pytorch_lightning as pl
import torch as th


def grad(model: pl.LightningModule, x: th.Tensor) -> th.Tensor:
    """
    Vanilla gradient method.

    Args:
        model (pl.LightningModule): The model to interpret.
        x (th.Tensor): Input data.

    Returns:
        th.Tensor: Gradients.

    References:
        https://arxiv.org/pdf/1312.6034.pdf
    """
    # Set training mode to compute gradients
    model.train()

    # Get outputs
    x = x.detach().requires_grad_(True)
    out = model(x)

    # Clear gradients
    model.zero_grad()

    # Compute gradients
    score_max_index = out.argmax().item()
    score_max = out[0, score_max_index]
    score_max.backward()

    return x.grad.detach().cpu()
