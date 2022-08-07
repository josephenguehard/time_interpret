from torch import Tensor
from typing import Tuple, Union


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
