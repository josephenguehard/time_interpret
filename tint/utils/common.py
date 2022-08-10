import copy

import torch

from captum._utils.common import (
    _expand_additional_forward_args,
    _format_baseline,
    _format_input,
    _format_output,
    _format_additional_forward_args,
    _run_forward,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from torch import Tensor
from typing import Any, Callable, Tuple


def _add_temporal_mask(
    inputs: Tensor,
    target: Tensor = None,
    additional_forward_args: Any = None,
    temporal_target: bool = False,
    temporal_additional_forward_args: Tuple[bool] = None,
):
    # Format input data
    inputs = _format_input(inputs)
    additional_forward_args = _format_additional_forward_args(
        additional_forward_args
    )

    # Inputs is only of length one
    data = inputs[0]

    # Create a lower triangular mask
    temporal_mask = torch.ones((data.shape[0], data.shape[1], data.shape[1]))
    temporal_mask = torch.tril(temporal_mask)
    temporal_mask = temporal_mask.reshape(
        (data.shape[0] * data.shape[1], data.shape[1])
        + (1,) * len(data.shape[2:])
    )

    # Expand data and args along the first dim
    data = torch.cat([data] * data.shape[1], dim=0)
    additional_forward_args = _expand_additional_forward_args(
        additional_forward_args, data.shape[1]
    )
    if target is not None:
        target = torch.cat([target] * data.shape[1], dim=0)

    # Multiply data and args by the tempora mask
    data = data * temporal_mask
    if additional_forward_args is not None:
        additional_forward_args = tuple(
            arg * temporal_mask if is_temporal else arg
            for arg, is_temporal in zip(
                additional_forward_args,
                temporal_additional_forward_args,
            )
        )
    if target is not None and temporal_target:
        target = target * temporal_target

    return data, additional_forward_args, target


def _slice_to_time(
    inputs: TensorOrTupleOfTensorsGeneric,
    time: int,
    *args,
    forward_func: Callable = None,
    task: str = "none",
    threshold: float = 0.5,
    temporal_target: bool = False,
    temporal_additional_forward_args: Tuple[bool] = None,
    **kwargs,
):
    assert task in [
        "none",
        "binary",
        "multilabel",
        "multiclass",
        "regression",
    ], "task is not recognised."
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    # Format inputs
    is_inputs_tuple = isinstance(inputs, tuple)
    inputs = _format_input(inputs)

    # Slice inputs
    partial_inputs = tuple(x[:, :time, ...] for x in inputs)

    # Format and slice args
    args_copy = copy.deepcopy(args)
    is_args_tuple = tuple(isinstance(arg, tuple) for arg in args_copy)
    args_copy = tuple(_format_input(arg) for arg in args_copy)
    args_copy = tuple(
        tuple(x[:, :time, ...] for x in arg) for arg in args_copy
    )

    kwargs_copy = copy.deepcopy(kwargs)

    # Format and slice baselines
    if "baselines" in kwargs_copy:
        baselines = _format_baseline(kwargs_copy["baselines"], partial_inputs)
        kwargs_copy["baselines"] = tuple(
            x[:, :time, ...] if isinstance(x, Tensor) else x for x in baselines
        )

    # Format and slice target
    if "target" in kwargs_copy and temporal_target:
        assert isinstance(
            kwargs_copy["target"], Tensor
        ), "target must be a tensor if temporal"
        kwargs_copy["target"] = kwargs_copy["target"][:, :time, ...]

    # Format and slice additional forward args
    if temporal_additional_forward_args is not None:
        additional_forward_args = _format_additional_forward_args(
            kwargs_copy["additional_forward_args"]
        )
        assert len(additional_forward_args) == len(
            temporal_additional_forward_args
        ), (
            "Length mismatch between additional_forward_args "
            "and temporal_additional_forward_args"
        )

        kwargs_copy["additional_forward_args"] = tuple(
            arg[:, :time, ...] if is_temporal else arg
            for arg, is_temporal in zip(
                additional_forward_args,
                temporal_additional_forward_args,
            )
        )

    # If forward_func is provided, compute partial predictions
    # according to task
    if forward_func is not None and task != "none":

        # Get additional args
        additional_forward_args = None
        if "additional_forward_args" in kwargs_copy:
            additional_forward_args = kwargs_copy["additional_forward_args"]

        with torch.autograd.set_grad_enabled(False):
            # Get model outputs
            partial_targets = _run_forward(
                forward_func,
                partial_inputs,
                additional_forward_args=additional_forward_args,
            )

        # Get target as predictions
        if task in ["binary", "multiclass"]:
            partial_targets = torch.argmax(partial_targets, -1)
        elif task == "multilabel":
            partial_targets = (
                torch.sigmoid(partial_targets) > threshold
            ).long()
            partial_targets = tuple(
                partial_targets[..., i]
                for i in range(partial_targets.shape[-1])
            )

        kwargs_copy["target"] = partial_targets

    return (
        _format_output(is_inputs_tuple, partial_inputs),
        *(
            _format_output(is_tuple, arg)
            for is_tuple, arg in zip(is_args_tuple, args_copy)
        ),
        kwargs_copy,
    )


def _validate_input(
    inputs: Tuple[Tensor, ...],
    data: Tuple[Tensor, ...],
    is_temporal: bool = False,
) -> None:
    assert len(inputs) == len(data), (
        "Input and data must have the same "
        "dimensions, data has {} features whereas input has {}.".format(
            len(data), len(inputs)
        )
    )

    for input, d in zip(inputs, data):
        if is_temporal:
            assert input.shape[2:] == d.shape[2:], (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found data: {} and input: {} ".format(d.shape, input.shape)
            )
        else:
            assert input.shape[1:] == d.shape[1:], (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found data: {} and input: {} ".format(d.shape, input.shape)
            )
