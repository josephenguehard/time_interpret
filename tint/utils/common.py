import torch

from captum._utils.common import (
    _expand_additional_forward_args,
    _format_input,
    _format_additional_forward_args,
)

from torch import Tensor
from typing import Any, Tuple


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
