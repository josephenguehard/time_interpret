from captum.attr._utils.attribution import GradientAttribution

from typing import Callable


class GeodesicIntegratedGradients(GradientAttribution):
    """
    Geodesic Integrated Gradients.

    Args:
        forward_func (callable):  The forward function of the model or any
            modification of it

    """

    def __init__(self, forward_func: Callable):
        super().__init__(forward_func=forward_func)
