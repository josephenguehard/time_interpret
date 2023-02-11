import torch
import typing

from captum.attr import IntegratedGradients
from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_inputs,
    _format_additional_forward_args,
    _format_baseline,
    _is_tuple,
    _format_tensor_into_tuples,
    _format_output,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import _reshape_and_sum

from tint.utils import get_progress_bars, _slice_to_time

from torch import Tensor
from typing import Any, Callable, List, Tuple, Union, cast


class TemporalIntegratedGradients(IntegratedGradients):
    """
    Temporal Integrated Gradients.

    This method computes gradients iteratively on a time series as such:
    it crops the sequence up to a time, and only moves this last time from
    a baseline to its original value.

    The number of steps per time depends on the strategy. If it is
    ``'fixed'``, then n_steps gradients are computed for each time.
    If it is ``'interval'``, the number of steps depends on the interval
    between two times: the larger, the greater number of points.

    Args:
        forward_func (callable): The forward function of the model or
            any modification of it.
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

    Examples:
        >>> import torch as th
        >>> from tint.attr import TemporalIntegratedGradients
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = TemporalIntegratedGradients(mlp)
        >>> attr = explainer.attribute(inputs, target=0)
    """

    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
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
        times: Tensor = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        strategy: str = "fixed",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        temporal_target: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        times: Tensor = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        strategy: str = "fixed",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True] = True,
        temporal_target: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        show_progress: bool = False,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        times: Tensor = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        strategy: str = "fixed",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        temporal_target: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        show_progress: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        """
        Attribute method.

        Args:
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
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
            times (Tensor, optional): Tensor of times. If not provided, it is
                assumed that the points are temporally equally spaced.
                Default: None
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
            method (string, optional): Method for approximating the integral,
                one of `riemann_right`, `riemann_left`, `riemann_middle`,
                `riemann_trapezoid` or `gausslegendre`.
                Default: `gausslegendre` if no method is provided.
            strategy (str, optinal): Strategy to distribute gradients
                evaluations over time. Either ``'fixed'`` or ``'interval'``
                Default: 'fixed'
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
            temporal_target: Temporal target. Default: False
            temporal_additional_forward_args (tuple): Set each
                additional forward arg which is temporal.
                Only used with return_temporal_attributions.
                Default: None
            return_temporal_attributions (bool): Whether to return all saliencies
                for all time points or only the last one per time point.
                Default: False
            show_progress (bool, optional): Displays the progress of
                computation. It will try to use tqdm if available for
                advanced features (e.g. time estimation). Otherwise, it
                will fallback to a simple output of progress.
                Default: False

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Integrated gradients with respect to each input feature.
                attributions will always be the same size as the provided
                inputs, with each value providing the attribution of the
                corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                The difference between the total approximated and true
                integrated gradients. This is computed using the property
                that the total sum of forward_func(inputs) -
                forward_func(baselines) must equal the total sum of the
                integrated gradient.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                of examples in inputs.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_inputs(inputs)

        # Get baselines
        baselines = _format_baseline(baselines, inputs)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Get n_steps depending on strategy
        n_steps = [n_steps] * inputs[0].shape[1]
        assert strategy in [
            "fixed",
            "interval",
        ], f"strategy must be either fixed or interval, got {strategy}."
        if strategy == "interval" and times is not None:
            max_interval = (times[1:] - times[:-1]).max().item()
            n_steps = [
                torch.div(t, max_interval / n_steps[0], rounding_mode="floor")
                .long()
                .item()
                + 1
                for t in times[1:] - times[:-1]
            ]

        attributions_partial_list = list()
        delta_partial_list = list()
        is_attrib_tuple = True

        times = range(1, inputs[0].shape[1] + 1)
        if show_progress:
            times = get_progress_bars()(
                times, desc=f"{self.get_name()} attribution"
            )

        for time in times:
            # Agg args into a kwarg dict
            kwargs = dict()
            kwargs["baselines"] = baselines
            kwargs["target"] = target
            kwargs["additional_forward_args"] = additional_forward_args

            # Slice data up to time
            partial_inputs, kwargs_copy = _slice_to_time(
                inputs=inputs,
                time=time,
                temporal_target=temporal_target,
                temporal_additional_forward_args=temporal_additional_forward_args,
                **kwargs,
            )

            # Recover partial data
            partial_baselines = kwargs_copy["baselines"]
            partial_target = kwargs_copy["target"]
            partial_additional_forward_args = kwargs_copy[
                "additional_forward_args"
            ]

            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = self.compute_partial_attribution(
                partial_inputs=partial_inputs,
                is_inputs_tuple=is_inputs_tuple,
                baselines=partial_baselines,
                target=partial_target,
                additional_forward_args=partial_additional_forward_args,
                n_steps=n_steps[time - 1],
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=return_convergence_delta,
            )
            attributions_partial_list.append(attributions_partial)
            delta_partial_list.append(delta_partial)

        attributions = tuple()
        if return_temporal_attributions:
            for i in range(len(attributions_partial_list[0])):
                attr = [
                    torch.zeros_like(
                        attributions_partial_list[-1][i],
                        dtype=attributions_partial_list[-1][i].dtype,
                    )
                    for _ in range(len(attributions_partial_list))
                ]
                for j in range(len(attributions_partial_list)):
                    attr[j][:, : j + 1, ...] = attributions_partial_list[j][i]
                attributions += (torch.stack(attr, dim=1),)
        else:
            for i in range(len(attributions_partial_list[0])):
                attributions += (
                    torch.stack(
                        [x[i][:, -1, ...] for x in attributions_partial_list],
                        dim=1,
                    ),
                )

        delta = None
        if return_convergence_delta:
            delta = torch.cat(delta_partial_list, dim=0)

        return self._apply_checks_and_return_attributions(
            attributions,
            is_attrib_tuple,
            return_convergence_delta,
            delta,
        )

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
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

        # scale features and compute gradients.
        # (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = self.scale_features(
            inputs=inputs,
            baselines=baselines,
            alphas=alphas,
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional
        # forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
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

    @staticmethod
    def scale_features(
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        alphas: List[float],
    ) -> Tuple[Tensor]:
        # Get last time of baseline if Tensor
        baselines = tuple(
            baseline[:, -1, ...] if isinstance(baseline, Tensor) else baseline
            for baseline in baselines
        )

        # Only rescale the last time of the inputs
        scaled_features_tpl = tuple(
            torch.cat(
                [
                    torch.cat(
                        [
                            input[:, :-1, ...],
                            (
                                baseline
                                + alpha * (input[:, -1, ...] - baseline)
                            ).unsqueeze(1),
                        ],
                        dim=1,
                    )
                    for alpha in alphas
                ],
                dim=0,
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        return cast(Tuple[Tensor, ...], scaled_features_tpl)

    def compute_partial_attribution(
        self,
        partial_inputs: Tuple[Tensor, ...],
        is_inputs_tuple: bool,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Tuple[Tuple[Tensor, ...], bool, Union[None, Tensor]]:
        attributions = super().attribute.__wrapped__(
            self,
            partial_inputs if is_inputs_tuple else partial_inputs[0],
            baselines=baselines if is_inputs_tuple else baselines[0],
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=return_convergence_delta,
        )
        delta = None

        if return_convergence_delta:
            attributions, delta = attributions

        is_attrib_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)

        return (
            cast(Tuple[Tensor, ...], attributions),
            cast(bool, is_attrib_tuple),
            delta,
        )

    @staticmethod
    def _apply_checks_and_return_attributions(
        attributions: Tuple[Tensor, ...],
        is_attrib_tuple: bool,
        return_convergence_delta: bool,
        delta: Union[None, Tensor],
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        attributions = _format_output(is_attrib_tuple, attributions)

        ret = (
            (attributions, cast(Tensor, delta))
            if return_convergence_delta
            else attributions
        )
        ret = cast(
            Union[
                TensorOrTupleOfTensorsGeneric,
                Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
            ],
            ret,
        )
        return ret
