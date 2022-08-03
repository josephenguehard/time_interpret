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
        (
            attr * np.abs(np.log2(EPS + attr))
            + (1 - attr) * np.abs(np.log2(EPS + 1 - attr))
        ).sum()
        for attr in attributions_subset
    )
    return cast(Tuple[float, ...], ent)


@log_usage()
def entropy(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
) -> Tuple[float]:
    return _base_white_box_metric(
        metric=_entropy,
        attributions=attributions,
        true_attributions=true_attributions,
    )
