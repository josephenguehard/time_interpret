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
    pre_rec_tpl = tuple(
        precision_recall_curve(true_attr, attr)
        for true_attr, attr in zip(true_attributions, attributions)
    )
    return tuple(auc(pre_rec[2], pre_rec[0][:-1]) for pre_rec in pre_rec_tpl)


@log_usage()
def aup(
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
) -> Tuple[float]:
    return _base_white_box_metric(
        metric=_aup,
        attributions=attributions,
        true_attributions=true_attributions,
        normalize=normalize,
    )
