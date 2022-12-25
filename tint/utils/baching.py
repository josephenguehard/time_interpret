import copy
import torch as th

from captum._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_target,
)

from torch import Tensor
from typing import Tuple

from .tqdm import get_progress_bars


def _geodesic_batch_attribution(
    attr_method,
    inputs: Tuple[Tensor, ...],
    idx: Tuple[Tensor, ...],
    knns: Tuple[Tensor, ...],
    internal_batch_size: int,
    show_progress: bool = False,
    **kwargs,
):
    """
    This method applies internal batching to given geodesic attribution
    method, dividing the total steps into batches and running each
    independently and sequentially, stacking each result to compute the
    total grads and grads norm.

    kwargs include all argument necessary to pass to each attribute call.
    """
    grads_norm = None
    total_grads = None

    steps = range(0, len(idx[0]), internal_batch_size)
    if show_progress:
        steps = get_progress_bars()(
            steps,
            desc=f"Geodesic Integrated Gradients attribution",
        )

    for i in steps:
        # Get partial inputs and baselines
        partial_knns = tuple(knn[i : i + internal_batch_size] for knn in knns)
        partial_idx = tuple(id[i : i + internal_batch_size] for id in idx)

        # Expand and update additional args
        n_samples = len(partial_knns[0])
        kwargs_copy = copy.deepcopy(kwargs)
        _expand_and_update_additional_forward_args(
            n_samples=n_samples, kwargs=kwargs_copy
        )
        _expand_and_update_target(
            n_samples=n_samples,
            kwargs=kwargs_copy,
        )

        # Compute gradients
        partial_grads_norm, partial_grads = attr_method._attribute(
            inputs=tuple(x[knn] for x, knn in zip(inputs, partial_knns)),
            baselines=tuple(x[id] for x, id in zip(inputs, partial_idx)),
            **kwargs_copy,
        )

        # Save and stack outputs
        if grads_norm is None:
            grads_norm = partial_grads_norm
            total_grads = partial_grads
        else:
            grads_norm = tuple(
                th.cat([x, y], dim=0)
                for x, y in zip(grads_norm, partial_grads_norm)
            )
            total_grads = tuple(
                th.cat([x, y], dim=0)
                for x, y in zip(total_grads, partial_grads)
            )

    return grads_norm, total_grads
