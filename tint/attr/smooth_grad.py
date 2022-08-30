from captum.attr import Saliency, NoiseTunnel
from typing import Callable


def SmoothGrad(forward_func: Callable):
    """
    Wrapper combining Saliency and NoiseTunnel to recover SmoothGrad.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it.
    """
    return NoiseTunnel(Saliency(forward_func=forward_func))
