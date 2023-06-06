import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from typing import Tuple, cast

from .base import _base_white_box_metric

EPS = 1e-5


def _entropy(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    ent = tuple(
        float(
            (
                attr * np.abs(np.log2(EPS + attr))
                + (1 - attr) * np.abs(np.log2(EPS + 1 - attr))
            ).sum()
        )
        for attr in attributions_subset
    )
    return cast(Tuple[float, ...], ent)


@log_usage()
def entropy(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    """
    Entropy measure of the attributions over the true_attributions.

    This metric measures how much information there is in the attributions.
    A low entropy means that the attributions provide a lot of information.
    Lower is better.

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
        `Explaining Time Series Predictions with Dynamic Masks <https://arxiv.org/abs/2106.05303>`_

    Examples:
        >>> import torch as th
        >>> from tint.metrics.white_box import entropy
        <BLANKLINE>
        >>> attr = th.rand(8, 7, 5)
        >>> true_attr = th.randint(2, (8, 7, 5))
        <BLANKLINE>
        >>> entropy_ = entropy(attr, true_attr)
    """
    return _base_white_box_metric(
        metric=_entropy,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
        hard_labels=True,
    )
