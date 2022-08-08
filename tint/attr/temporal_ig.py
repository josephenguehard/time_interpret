import torch
import typing

from captum.attr import IntegratedGradients
from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_input,
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

from tint.utils import get_progress_bars

from torch import Tensor
from typing import Any, Callable, List, Tuple, Union, cast


class TemporalIntegratedGradients(IntegratedGradients):
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
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        temporal_additional_forward_args: Tuple[bool] = None,
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
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
        temporal_additional_forward_args: Tuple[bool] = None,
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
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        show_progress: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_input(inputs)

        # Get baselines
        baselines = _format_baseline(baselines, inputs)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        attributions_partial_list = list()
        delta_partial_list = list()
        is_attrib_tuple = True

        times = range(1, inputs[0].shape[1] + 1)
        if show_progress:
            times = get_progress_bars()(
                times, desc=f"{self.get_name()} attribution"
            )

        for time in times:
            # Get partial inputs up to time
            partial_inputs = tuple(x[:, :time, ...] for x in inputs)

            # Get partial baselines up to time if provided
            partial_baselines = tuple(
                x[:, :time, ...] if isinstance(x, Tensor) else x
                for x in baselines
            )

            # Get partial additional forward args if provided
            partial_additional_forward_args = additional_forward_args
            if temporal_additional_forward_args is not None:
                assert len(additional_forward_args) == len(
                    temporal_additional_forward_args
                ), (
                    "Length mismatch between additional_forward_args "
                    "and temporal_additional_forward_args"
                )
                partial_additional_forward_args = tuple(
                    x[:, :time, ...] if y else x
                    for x, y in zip(
                        additional_forward_args,
                        temporal_additional_forward_args,
                    )
                )

            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = self.compute_partial_attribution(
                partial_inputs=partial_inputs,
                is_inputs_tuple=is_inputs_tuple,
                baselines=partial_baselines,
                target=target,
                additional_forward_args=partial_additional_forward_args,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=return_convergence_delta,
            )
            attributions_partial_list.append(attributions_partial)
            delta_partial_list.append(delta_partial)

        attributions = tuple()
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
        # Only rescale the last time of the inputs
        scaled_features_tpl = tuple(
            torch.cat(
                [
                    torch.cat(
                        [
                            input[:, :-1, ...],
                            (
                                baseline[:, -1, ...]
                                + alpha
                                * (input[:, -1, ...] - baseline[:, -1, ...])
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

        return scaled_features_tpl

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
