from captum.log import log_usage
from captum._utils.common import _select_targets
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple

from .base import _base_metric


def _accuracy(
    prob_original: Tensor,
    prob_pert: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    assert 0.0 <= threshold <= 1.0, "threshold must be between 0 and 1"
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
    weight_fn: Callable[
        [Tuple[Tensor, ...], Tuple[Tensor, ...]], Tensor
    ] = None,
    threshold: float = 0.5,
) -> float:
    """
    Accuracy metric.

    This metric measures by how much the accuracy of a model drops when
    removing the topk most important features. Lower is better.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        inputs (tensor or tuple of tensors):  Input for which occlusion
            attributions are computed. If forward_func takes a single
            tensor as input, a single input tensor should be provided.
            If forward_func takes multiple tensors as input, a tuple
            of the input tensors should be provided. It is assumed
            that for all given input tensors, dimension 0 corresponds
            to the number of examples (aka batch size), and if
            multiple input tensors are provided, the examples must
            be aligned appropriately.
        attributions (tensor or tuple of tensors):
            The attributions with respect to each input feature.
            Attributions will always be
            the same size as the provided inputs, with each value
            providing the attribution of the corresponding input index.
            If a single tensor is provided as inputs, a single tensor
            is returned. If a tuple is provided for inputs, a tuple of
            corresponding sized tensors is returned.
        baselines (scalar, tensor, tuple of scalars or tensors, optional):
            Baselines define the starting point from which integral
            is computed and can be provided as:

            - a single tensor, if inputs is a single tensor, with
              exactly the same dimensions as inputs or the first
              dimension is one and the remaining dimensions match
              with inputs.

            - a single scalar, if inputs is a single tensor, which will
              be broadcasted for each input value in input tensor.

            - a tuple of tensors or scalars, the baseline corresponding
              to each tensor in the inputs' tuple can be:

              - either a tensor with matching dimensions to
                corresponding tensor in the inputs' tuple
                or the first dimension is one and the remaining
                dimensions match with the corresponding
                input tensor.

              - or a scalar, corresponding to a tensor in the
                inputs' tuple. This scalar value is broadcasted
                for corresponding input tensor.

            In the cases when `baselines` is not provided, we internally
            use zero scalar corresponding to each input tensor.

            Default: None
        additional_forward_args (any, optional): If the forward function
            requires additional arguments other than the inputs for
            which attributions should not be computed, this argument
            can be provided. It must be either a single additional
            argument of a Tensor or arbitrary (non-tuple) type or a
            tuple containing multiple additional arguments including
            tensors or any arbitrary python types. These arguments
            are provided to forward_func in order following the
            arguments in inputs.
            For a tensor, the first dimension of the tensor must
            correspond to the number of examples. It will be
            repeated for each of `n_steps` along the integrated
            path. For all other types, the given argument is used
            for all forward evaluations.
            Note that attributions are not computed with respect
            to these arguments.
            Default: None
        target (int, tuple, tensor or list, optional):  Output indices for
            which gradients are computed (for classification cases,
            this is usually the target class).
            If the network returns a scalar value per example,
            no target index is necessary.
            For general 2D outputs, targets can be either:

            - a single integer or a tensor containing a single
              integer, which is applied to all input examples

            - a list of integers or a 1D tensor, with length matching
              the number of examples in inputs (dim 0). Each integer
              is applied as the target for the corresponding example.

            For outputs with > 2 dimensions, targets can be either:

            - A single tuple, which contains #output_dims - 1
              elements. This target index is applied to all examples.

            - A list of tuples with length equal to the number of
              examples in inputs (dim 0), and each tuple containing
              #output_dims - 1 elements. Each tuple is applied as the
              target for the corresponding example.

            Default: None
        topk: Proportion of input to be dropped. Must be between 0 and 1.
            Default: 0.2
        weight_fn (Callable): Function to compute metrics weighting using
            original inputs and pertubed inputs. None if note provided.
            Default: None
        threshold: Threshold for the accuracy. Data higher than the threshold
            is considered as positive, and lower (strictly) negative.
            Default: 0.5

    Returns:
        (float): The accuracy metric.

    References:
        https://arxiv.org/pdf/2106.05303

    Examples:
        >>> import torch as th
        >>> from captum.attr import Saliency
        >>> from tint.metrics import accuracy
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = Saliency(mlp)
        >>> attr = explainer.attribute(inputs, target=0)
        <BLANKLINE>
        >>> acc = accuracy(mlp, inputs, attr, target=0)
    """
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
        weight_fn=weight_fn,
        threshold=threshold,
    )
