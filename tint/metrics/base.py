import torch

from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
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
from typing import Any, Callable, Tuple, Union, cast

from tint.utils import add_noise_to_inputs, _expand_baselines


@log_usage()
@torch.no_grad()
def _base_metric(
    metric: Callable,
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    n_samples: int = 1,
    n_samples_batch_size: int = None,
    stdevs: Union[float, Tuple[float, ...]] = 0.0,
    draw_baseline_from_distrib: bool = False,
    topk: float = 0.2,
    largest: bool = True,
    weight_fn: Callable[
        [Tuple[Tensor, ...], Tuple[Tensor, ...]], Tensor
    ] = None,
    classification: bool = True,
    **kwargs,
) -> float:
    # Format data
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    additional_forward_args = _format_additional_forward_args(
        additional_forward_args
    )
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore
    if baselines is not None:
        baselines = _format_baseline(
            baselines, cast(Tuple[Tensor, ...], inputs)
        )
        _validate_input(
            inputs,
            baselines,
            draw_baseline_from_distrib=draw_baseline_from_distrib,
        )

    # Set n_samples_batch_size to n_samples if not provided
    if n_samples_batch_size is None:
        n_samples_batch_size = n_samples

    # Loop over batches
    out = list()
    weights = list() if weight_fn else None
    for i in range(n_samples // n_samples_batch_size + 1):
        if n_samples - i * n_samples_batch_size >= n_samples_batch_size:
            n = n_samples_batch_size
        elif n_samples - i * n_samples_batch_size > 0:
            n = n_samples - i * n_samples_batch_size
        else:
            break

        # Add noise to inputs and expand
        _inputs = add_noise_to_inputs(inputs, stdevs, n)

        # Expand baselines
        _baselines = None
        if baselines is not None:
            _baselines = _expand_baselines(
                _inputs,
                baselines,
                n,
                draw_baseline_from_distrib,
            )

        # Expand additional args
        _additional_forward_args = _expand_additional_forward_args(
            additional_forward_args, n
        )

        # Expand target
        _target = _expand_target(target, n)

        # Format and expand attributions
        _attributions = _expand_additional_forward_args(attributions, n)

        # Validate topk
        assert 0 < topk < 1, "topk must be a float between 0 and 1"

        # Reverse topk is not largest to select the non topk
        if not largest:
            topk = 1.0 - topk

        # Clone inputs
        inputs_pert = tuple(inp.detach().clone() for inp in _inputs)

        # Get topk indices for each element in the batch
        # It is assumed that the batch is on the first dimension
        topk_indices = tuple(
            torch.topk(
                attr.reshape(len(attr), -1),
                int(attr.reshape(len(attr), -1).shape[-1] * topk),
                sorted=False,
                largest=largest,
            ).indices.to(attr.device)
            for attr in _attributions
        )

        # Set topk indices to inputs device
        topk_indices = tuple(
            topk.to(input.device) for topk, input in zip(topk_indices, _inputs)
        )

        # Replace topk values with baseline
        if _baselines is None:
            inputs_pert = tuple(
                inp.reshape(len(inp), -1)
                .scatter(-1, topk_idx, 0)
                .reshape(inp.shape)
                for inp, topk_idx in zip(inputs_pert, topk_indices)
            )
        else:
            _baselines = tuple(
                (
                    baseline
                    if isinstance(baseline, (int, float))
                    else baseline.reshape(len(baseline), -1).gather(
                        -1, topk_idx
                    )
                )
                for baseline, topk_idx in zip(_baselines, topk_indices)
            )
            inputs_pert = tuple(
                inp.reshape(len(inp), -1)
                .scatter(-1, topk_idx, baseline)
                .reshape(inp.shape)
                for inp, baseline, topk_idx in zip(
                    inputs_pert, _baselines, topk_indices
                )
            )

        # Get weights if provided
        if weight_fn:
            weights.append(weight_fn(_inputs, inputs_pert))

        # Get original predictions
        logits_original = _run_forward(
            forward_func=forward_func,
            inputs=_inputs,
            target=None,
            additional_forward_args=_additional_forward_args,
        )
        prob_original = logits_original.softmax(-1)

        # Get predictions for perturbed inputs
        logits_pert = _run_forward(
            forward_func=forward_func,
            inputs=inputs_pert,
            target=None,
            additional_forward_args=_additional_forward_args,
        )
        prob_pert = logits_pert.softmax(-1)

        # Get target as original predictions if not provided
        if _target is None and classification:
            _target = logits_original.argmax(-1)

        if classification:
            out.append(metric(prob_original, prob_pert, _target, **kwargs))
        else:
            out.append(metric(logits_original, logits_pert, _target, **kwargs))

    # Concat results
    out = torch.cat(out)

    # If weight_fn provided return a weighted average
    if weight_fn:
        weights = torch.cat(weights)
        weights = weights.to(out.device)
        return (out * weights).sum().item() / weights.sum().item()

    # Otherwise return the average
    return out.mean().item()
