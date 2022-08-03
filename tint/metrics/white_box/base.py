import torch

from captum.log import log_usage
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from typing import Callable, Tuple

EPS = 1e-5


@log_usage()
def _base_white_box_metric(
    metric: Callable,
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
) -> Tuple[float]:
    # Convert attributions into tuple
    attributions = _format_tensor_into_tuples(attributions)
    true_attributions = _format_tensor_into_tuples(true_attributions)

    # Take the absolute value of attributions
    attributions = tuple(torch.abs(attr) for attr in attributions)

    # Normalise attributions
    min_tpl = tuple(
        attr.reshape(len(attr), -1).min(dim=-1).values for attr in attributions
    )
    max_tpl = tuple(
        attr.reshape(len(attr), -1).max(dim=-1).values for attr in attributions
    )
    attributions = tuple(
        (attr - min_) / (max_ + EPS)
        for attr, min_, max_ in zip(attributions, min_tpl, max_tpl)
    )

    # Reshape attributions and get a subset corresponding to the true ones
    attributions = tuple(attr.reshape(-1) for attr in attributions)
    true_attributions = tuple(
        attr.reshape(-1).int() for attr in true_attributions
    )
    attributions_subset = tuple(
        attr[true_attr != 0]
        for attr, true_attr in zip(attributions, true_attributions)
    )

    # Convert to numpy
    attributions = tuple(attr.numpy() for attr in attributions)
    true_attributions = tuple(attr.numpy() for attr in true_attributions)
    attributions_subset = tuple(attr.numpy() for attr in attributions_subset)

    # Compute metric
    return metric(attributions, true_attributions, attributions_subset)
