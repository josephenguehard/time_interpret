import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.metrics import mean_squared_error
from typing import Tuple, cast

from .base import _base_white_box_metric


def _rmse(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    rmse_tpl = tuple(
        mean_squared_error(true_attr, attr, squared=False)
        for true_attr, attr in zip(true_attributions, attributions)
    )
    return cast(Tuple[float], rmse_tpl)


@log_usage()
def rmse(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    """
    Root mean squared error.

    This is the standard root mean squared error. Lower is better.

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
        (float or tuple or floats): The aup metric.

    Examples:
        >>> import torch as th
        >>> from tint.metrics.white_box import aup
        <BLANKLINE>
        >>> attr = th.rand(8, 7, 5)
        >>> true_attr = th.randint(2, (8, 7, 5))
        <BLANKLINE>
        >>> aup_ = aup(attr, true_attr)
    """
    return _base_white_box_metric(
        metric=_rmse,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
        hard_labels=False,
    )
