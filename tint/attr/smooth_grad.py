from captum.attr import Saliency, NoiseTunnel
from typing import Callable


def SmoothGrad(forward_func: Callable):
    """
    Wrapper combining Saliency and NoiseTunnel to recover SmoothGrad.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it.

    Examples:
        >>> import torch as th
        >>> from tint.attr import SmoothGrad
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = SmoothGrad(mlp)
        >>> attr = explainer.attribute(inputs, target=0)
    """
    return NoiseTunnel(Saliency(forward_func=forward_func))
