#!/usr/bin/env python3
import math
import typing
from typing import Any, Callable, List, Tuple, Union
import warnings

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr import IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor

EPSILON = 1e-9


class GuidedIntegratedGradients(IntegratedGradients):
    r"""
    Guided Integrated Gradients.

    This method greedily search for a path which avoids high-gradients regions.
    For this, it selects a ``fraction`` of the data corresponding to the lowest
    gradients, and move this subset of the current point on the path closer to
    the input data, starting from a baseline. This process is repeated
    ``n_guided_steps`` times. Moreover, it anchors the resulting path to the
    straight line between inputs and baselines by forcing the path to cross
    points on this line. The number of anchor points is controlled by the
    ``n_anchors`` parameter.

    Similarly to IntegratedGradients, it is possible to provide an
    ``internal_batch_size`` to reduce memory usage. The convergence delta
    can also be returned.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it
        multiply_by_inputs (bool, optional): Indicates whether to factor
            model inputs' multiplier in the final attribution scores.
            In the literature this is also known as local vs global
            attribution. If inputs' multiplier isn't factored in,
            then that type of attribution method is also called local
            attribution. If it is, then that type of attribution
            method is called global.
            More detailed can be found here:
            https://arxiv.org/abs/1711.06104

            In case of integrated gradients, if `multiply_by_inputs`
            is set to True, final sensitivity scores are being multiplied by
            (inputs - baselines).

    References:
        `Guided Integrated Gradients: an Adaptive Path Method for Removing Noise <https://arxiv.org/abs/2106.09788>`
    """

    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
    ) -> None:
        IntegratedGradients.__init__(
            self,
            forward_func,
            multiply_by_inputs=multiply_by_inputs,
        )

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.
    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_anchors: int = 0,
        n_guided_steps: int = 50,
        fraction: float = 0.1,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_anchors: int = 0,
        n_guided_steps: int = 50,
        fraction: float = 0.1,
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_anchors: int = 0,
        n_guided_steps: int = 50,
        fraction: float = 0.1,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:
            inputs (Tensor or tuple[Tensor, ...]): Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
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
            target (int, tuple, Tensor, or list, optional): Output indices for
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
            additional_forward_args (Any, optional): If the forward function
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
            n_anchors (int, optional): The number of anchor points used by the
                method. Default: 0
            n_guided_steps (int, optional): The number of steps to compute the path
                between two anchor points. Default: 50
            fraction (float, optional): Fraction of features (we use 10%) with the
                lowest absolute gradient values to be selected by the algorithm.
                Default: 0.1
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. internal_batch_size must be at least equal to
                #examples.
                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain internal_batch_size / num_devices examples.
                If internal_batch_size is None, then all evaluations are
                processed in one batch.
                Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                convergence delta or not. If `return_convergence_delta`
                is set to True convergence delta will be returned in
                a tuple following attributions.
                Default: False

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Integrated gradients with respect to each input feature.
                attributions will always be the same size as the provided
                inputs, with each value providing the attribution of the
                corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                The difference between the total approximated and true
                integrated gradients. This is computed using the property
                that the total sum of forward_func(inputs) -
                forward_func(baselines) must equal the total sum of the
                integrated gradient.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in inputs.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        method = "riemann_trapezoid"
        n_anchors += 2  # Add inputs and baselines points as anchors
        _validate_input(inputs, baselines, n_anchors, method)

        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            if internal_batch_size < (2 * num_examples):
                warnings.warn(
                    "Internal batch size cannot be less than twice the number of input examples. "
                    "Defaulting to internal batch size of %d equal to twice the number of examples."
                    % (2 * num_examples)
                )
                internal_batch_size = 2 * num_examples
            attributions = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_anchors,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_guided_steps=n_guided_steps,
                fraction=fraction,
                method=method,
            )
        else:
            attributions = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_anchors,
                n_guided_steps=n_guided_steps,
                fraction=fraction,
                method=method,
            )

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_output(is_inputs_tuple, attributions), delta
        return _format_output(is_inputs_tuple, attributions)

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 2,
        n_guided_steps: int = 50,
        fraction: float = 0.1,
        method: str = "riemann_trapezoid",
        step_sizes_and_alphas: Union[
            None, Tuple[List[float], List[float]]
        ] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features, baselines and compute gradients.
        # (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * (#steps - 1) x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [
                    baseline + alpha * (input - baseline)
                    for alpha in alphas[1:]
                ],
                dim=0,
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )
        scaled_baselines_tpl = tuple(
            torch.cat(
                [
                    baseline + alpha * (input - baseline)
                    for alpha in alphas[:-1]
                ],
                dim=0,
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * (#steps - 1) x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args, n_steps - 1
            )
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps - 1)

        # grads: dim -> (bsz * (#steps - 1) x inputs[0].shape[1:], ...)
        # For each guided step, we update the baseline and compute
        # the corresponding gradients
        l1_total = self.l1(inputs, baselines)
        grads = tuple(
            torch.zeros_like(feature) for feature in scaled_features_tpl
        )
        for step in range(n_guided_steps):
            new_grads = self.gradient_func(
                forward_fn=self.forward_func,
                inputs=scaled_baselines_tpl,
                target_ind=expanded_target,
                additional_forward_args=input_additional_args,
            )
            new_grads, scaled_baselines_tpl = tuple(
                zip(
                    *(
                        self.accumulate_grads(
                            input=input,
                            baseline=baseline,
                            grad_in=grad,
                            step=step,
                            n_guided_steps=n_guided_steps,
                            l1_total=l1,
                            fraction=fraction,
                        )
                        for input, baseline, grad, l1 in zip(
                            scaled_features_tpl,
                            scaled_baselines_tpl,
                            new_grads,
                            l1_total,
                        )
                    )
                )
            )
            grads = tuple(
                grad + new_grad for grad, new_grad in zip(grads, new_grads)
            )

        # flattening grads so that we can multiply it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps - 1, -1)
            * torch.tensor(step_sizes[:-1])
            .view(n_steps - 1, 1)
            .to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad,
                n_steps - 1,
                grad.shape[0] // (n_steps - 1),
                grad.shape[1:],
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(
                    total_grads, inputs, baselines
                )
            )
        return attributions

    def accumulate_grads(
        self,
        input: Tensor,
        baseline: Tensor,
        grad_in: Tensor,
        step: int,
        n_guided_steps: int,
        l1_total: float,
        fraction: float = 0.1,
    ):
        gamma = torch.inf
        scaled_features = baseline.clone()
        grad_out = torch.zeros_like(grad_in)

        # The L1 target is the one we should have reached at the end
        # of this step
        l1_target = l1_total * (1 - (step + 1) / n_guided_steps)

        # We iterate until L1 target is reached
        while gamma > 1.0:
            # We ignore features which are already on the target
            grad_in[scaled_features == input] = torch.inf

            # Current L1, to be compared with the target
            l1_current = self.l1(scaled_features, input)

            # If the current L1 is close enough, break
            if math.isclose(
                l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON
            ):
                grad_out += (scaled_features - baseline) * grad_in
                break

            # We select which features have the lowest grads
            threshold = torch.quantile(
                grad_in.abs(), fraction, interpolation="lower"
            )
            s = torch.logical_and(
                grad_in.abs() <= threshold, grad_in != torch.inf
            )

            # We check how much the L1 can be reduced by updating
            # the selected features
            l1_s = ((scaled_features - input).abs() * s).sum()
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = torch.inf

            # If L1 target is reached, we update the features to correspond
            # to this target. Otherwise, we update all the selected features
            # and iterated until convergence
            if gamma > 1.0:
                scaled_features[s] = input[s]
            else:
                scaled_features[s] += gamma * (input[s] - scaled_features[s])

            # We remove introduced Inf
            grad_in[grad_in == torch.inf] = 0

            # We only accumulate gradients here, multiplying by the inputs
            # is done at the end of the `attribute` method
            grad_out[s] += grad_in[s]

        return grad_out, scaled_features

    @staticmethod
    def l1(
        x1: TensorOrTupleOfTensorsGeneric, x2: TensorOrTupleOfTensorsGeneric
    ) -> Union[int, Tuple[int]]:
        if isinstance(x1, tuple):
            return tuple(  # type:ignore
                (x - y).abs().sum().item() for x, y in zip(x1, x2)
            )
        return (x1 - x2).abs().sum().item()
