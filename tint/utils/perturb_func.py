import torch

from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from torch import Tensor
from typing import Tuple


def default_perturb_func(
    inputs: TensorOrTupleOfTensorsGeneric, perturb_radius: float = 0.02
) -> Tuple[Tensor, ...]:
    r"""A default function for generating perturbations of `inputs`
    within perturbation radius of `perturb_radius`.
    This function samples uniformly random from the L_Infinity ball
    with `perturb_radius` radius.
    The users can override this function if they prefer to use a
    different perturbation function.

    Args:

        inputs (tensor or a tuple of tensors): The input tensors that we'd
                like to perturb by adding a random noise sampled unifromly
                random from an L_infinity ball with a radius `perturb_radius`.

        perturb_radius (float): A radius used for sampling from
                an L_infinity ball.

    Returns:

        perturbed_input (tuple(tensor)): A list of perturbed inputs that
                are createed by adding noise sampled uniformly random
                from L_infiniy ball with a radius `perturb_radius` to the
                original inputs.

    """
    inputs = _format_tensor_into_tuples(inputs)
    perturbed_input = tuple(
        input
        + torch.FloatTensor(input.size())  # type: ignore
        .uniform_(-perturb_radius, perturb_radius)
        .to(input.device)
        for input in inputs
    )
    return perturbed_input
