import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.metrics import roc_auc_score
from typing import Tuple

from .base import _base_white_box_metric


def _roc_auc(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    return tuple(
        roc_auc_score(true_attr, attr)
        for true_attr, attr in zip(true_attributions, attributions)
    )


@log_usage()
def roc_auc(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
) -> Tuple[float]:
    return _base_white_box_metric(
        metric=_roc_auc,
        attributions=attributions,
        true_attributions=true_attributions,
    )
