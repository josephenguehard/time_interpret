import torch

from captum.log import log_usage
from captum._utils.common import (
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    _validate_input,
    _validate_target,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple, cast


@log_usage()
def log_odds(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    topk: float = 0.2,
):
    # perform argument formatting
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    if baselines is not None:
        baselines = _format_baseline(
            baselines, cast(Tuple[Tensor, ...], inputs)
        )
    additional_forward_args = _format_additional_forward_args(
        additional_forward_args
    )
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore

    # Validate inputs
    if baselines is not None:
        _validate_input(inputs, baselines)

    # Get original predictions
    logits_original = _run_forward(
        forward_func=forward_func,
        inputs=inputs,
        target=target,
        additional_forward_args=additional_forward_args,
    )

    # Clone inputs
    inputs = (inp.detach().clone() for inp in inputs)

    # Get topk indices
    topk_indices = (
        torch.topk(
            attr.reshape(-1), int(len(attr.reshape(-1)) * topk), sorted=False
        ).indices
        for attr in attributions
    )

    # Replace topk values with baseline
    if baselines is None:
        for inp, topk_idx in zip(inputs, topk_indices):
            inp.reshape(-1)[topk_idx] = 0
    else:
        for inp, baseline, topk_idx in zip(inputs, baselines, topk_indices):
            if isinstance(baseline, int, float):
                inp.reshape(-1)[topk_idx] = baseline
            else:
                inp.reshape(-1)[topk_idx] = baseline.reshape(-1)[topk_idx]
