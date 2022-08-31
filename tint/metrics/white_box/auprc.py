import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.metrics import average_precision_score
from typing import Tuple

from .base import _base_white_box_metric


def _auprc(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    return tuple(
        average_precision_score(true_attr, attr)
        for true_attr, attr in zip(true_attributions, attributions)
    )


@log_usage()
def auprc(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    """
    Area under precision-recall.

    This is the standard area under the precision-recall curve.
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
        (float or tuple or floats): The auprc metric.

    Examples:
        >>> import torch as th
        >>> from tint.metrics.white_box import auprc
        <BLANKLINE>
        >>> attr = th.rand(8, 7, 5)
        >>> true_attr = th.randint(2, (8, 7, 5))
        <BLANKLINE>
        >>> auprc_ = auprc(attr, true_attr)
    """
    return _base_white_box_metric(
        metric=_auprc,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
    )
