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
        (np.abs(np.log2(1 - attr + EPS))).sum() for attr in attributions_subset
    )
    return cast(Tuple[float, ...], info)


@log_usage()
def information(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    return _base_white_box_metric(
        metric=_information,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
    )
