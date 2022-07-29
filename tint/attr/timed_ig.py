import torch
import typing

from captum.attr import IntegratedGradients
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
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

from torch import Tensor
from typing import Any, Callable, Tuple, Union, cast


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
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_input(inputs)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        attributions_partial_list = list()
        delta_partial_list = list()
        is_attrib_tuple = True
        for time in range(1, inputs[0].shape[1] + 1):
            partial_inputs = tuple(x[:, :time, ...] for x in inputs)

            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = self.compute_partial_attribution(
                partial_inputs=partial_inputs,
                is_inputs_tuple=is_inputs_tuple,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
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
            baselines=baselines,
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
