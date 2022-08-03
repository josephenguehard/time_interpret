from captum.log import log_usage
from captum._utils.common import _select_targets
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable

from .base import _base_metric


def _accuracy(
    prob_original: Tensor,
    prob_pert: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    return (_select_targets(prob_pert, target) >= threshold).float()


@log_usage()
def accuracy(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    topk: float = 0.2,
    threshold: float = 0.5,
) -> Tensor:
    return _base_metric(
        metric=_accuracy,
        forward_func=forward_func,
        inputs=inputs,
        attributions=attributions,
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        target=target,
        topk=topk,
        largest=True,
        threshold=threshold,
    )
