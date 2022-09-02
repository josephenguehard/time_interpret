import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from typing import Tuple, cast

from .base import _base_white_box_metric

EPS = 1e-5


def _information(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    info = tuple(
        float((np.abs(np.log2(1 - attr + EPS))).sum())
        for attr in attributions_subset
    )
    return cast(Tuple[float, ...], info)


@log_usage()
def information(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    """
    Information measure of the attributions over the true_attributions.

    This metric measures how much information there is in the attributions.
    Higher is better.

    Args:
        attributions (tensor or tuple of tensors):
            The attributions with respect to each input feature.
            Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single float
            is returned. If a tuple is provided for inputs, a tuple of
            float is returned.
        true_attributions (tensor or tuple of tensors):
            True attributions to be used as a benchmark. Should be of
            the same format as the attributions.
        normalize (bool): Whether to normalize the attributions before
            computing the metric or not. Default: True

    Returns:
        (float or tuple or floats): The aur metric.

    References:
        https://arxiv.org/pdf/2106.05303

    Examples:
        >>> import torch as th
        >>> from tint.metrics.white_box import information
        <BLANKLINE>
        >>> attr = th.rand(8, 7, 5)
        >>> true_attr = th.randint(2, (8, 7, 5))
        <BLANKLINE>
        >>> information_ = information(attr, true_attr)
    """
    return _base_white_box_metric(
        metric=_information,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
    )
