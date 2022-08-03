import numpy as np

from captum.log import log_usage
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.metrics import precision_recall_curve, auc
from typing import Tuple

from .base import _base_white_box_metric


def _aup(
    attributions: Tuple[np.ndarray],
    true_attributions: Tuple[np.ndarray],
    attributions_subset: Tuple[np.ndarray],
) -> Tuple[float]:
    precision_tpl, recall_tpl, thresholds_tpl = tuple(
        precision_recall_curve(true_attr, attr)
        for true_attr, attr in zip(true_attributions, attributions)
    )
    return tuple(
        auc(thresholds, precision[:-1])
        for thresholds, precision in zip(thresholds_tpl, precision_tpl)
    )


@log_usage()
def aup(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
) -> Tuple[float]:
    return _base_white_box_metric(
        metric=_aup,
        attributions=attributions,
        true_attributions=true_attributions,
    )
