import torch

from captum.log import log_usage
from captum._utils.common import (
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    _validate_input,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple, cast


@log_usage()
def _base_metric(
    metric: Callable,
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    topk: float = 0.2,
    largest: bool = True,
    weight_fn: Callable[
        [Tuple[Tensor, ...], Tuple[Tensor, ...]], Tensor
    ] = None,
    classification: bool = True,
    **kwargs,
) -> float:
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

    # Validate topk
    assert 0 < topk < 1, "topk must be a float between 0 and 1"

    # Reverse topk is not largest to select the non topk
    if not largest:
        topk = 1.0 - topk

    # Clone inputs
    inputs_pert = tuple(inp.detach().clone() for inp in inputs)

    # Get topk indices for each element in the batch
    # It is assumed that the batch is on the first dimension
    topk_indices = tuple(
        torch.topk(
            attr.reshape(len(attr), -1),
            int(attr.reshape(len(attr), -1).shape[-1] * topk),
            sorted=False,
            largest=largest,
        ).indices.to(attr.device)
        for attr in attributions
    )

    # Set topk indices to inputs device
    topk_indices = tuple(
        topk.to(input.device) for topk, input in zip(topk_indices, inputs)
    )

    # Replace topk values with baseline
    if baselines is None:
        inputs_pert = tuple(
            inp.reshape(len(inp), -1)
            .scatter(-1, topk_idx, 0)
            .reshape(inp.shape)
            for inp, topk_idx in zip(inputs_pert, topk_indices)
        )
    else:
        baselines = tuple(
            baseline
            if isinstance(baseline, (int, float))
            else baseline.reshape(len(baseline), -1).gather(-1, topk_idx)
            for baseline, topk_idx in zip(baselines, topk_indices)
        )
        inputs_pert = tuple(
            inp.reshape(len(inp), -1)
            .scatter(-1, topk_idx, baseline)
            .reshape(inp.shape)
            for inp, baseline, topk_idx in zip(
                inputs_pert, baselines, topk_indices
            )
        )

    # Get weights if provided
    weights = None
    if weight_fn:
        weights = weight_fn(inputs, inputs_pert)

    # Get original predictions
    logits_original = _run_forward(
        forward_func=forward_func,
        inputs=inputs,
        target=None,
        additional_forward_args=additional_forward_args,
    )
    prob_original = logits_original.softmax(-1)

    # Get predictions for perturbed inputs
    logits_pert = _run_forward(
        forward_func=forward_func,
        inputs=inputs_pert,
        target=None,
        additional_forward_args=additional_forward_args,
    )
    prob_pert = logits_pert.softmax(-1)

    # Get target as original predictions if not provided
    if target is None and classification:
        target = logits_original.argmax(-1)

    if classification:
        out = metric(prob_original, prob_pert, target, **kwargs)
    else:
        out = metric(logits_original, logits_pert, target, **kwargs)

    if weight_fn:
        weights = weights.to(out.device)
        return (out * weights).sum().item() / weights.sum().item()

    return out.mean().item()
