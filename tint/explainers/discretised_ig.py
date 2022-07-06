from typing import Any, Callable, Tuple, Union

import torch
from torch import Tensor

from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import _reshape_and_sum, _format_input


class DiscretetizedIntegratedGradients(GradientAttribution):
    """
    Discretetized Integrated Gradients.

    Args:
        forward_func (callable):  The forward function of the model or any
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
        https://github.com/INK-USC/DIG
        https://arxiv.org/abs/2108.13654
    """
    def __init__(
        self, forward_func: Callable, multiply_by_inputs: bool = True
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    @log_usage()
    def attribute(
        self,
        scaled_features: Tuple[Tensor, ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(scaled_features)
        scaled_features_tpl = _format_input(scaled_features)

        attributions = self.calculate_dig_attributions(
            scaled_features_tpl=scaled_features_tpl,
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
        )
        if return_convergence_delta:
            assert (
                len(scaled_features_tpl) == 1
            ), "More than one tuple not supported in this code!"
            start_point, end_point = _format_input(
                scaled_features_tpl[0][0].unsqueeze(0)
            ), _format_input(
                scaled_features_tpl[0][-1].unsqueeze(0)
            )  # baselines, inputs (only works for one input, len(tuple) == 1)
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

    def calculate_dig_attributions(
        self,
        scaled_features_tpl: Tuple[Tensor, ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
    ) -> Tuple[Tensor, ...]:
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
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

        # calculate (x - x') for each interpolated point
        shifted_inputs_tpl = tuple(
            torch.cat([scaled_features[1:], scaled_features[-1].unsqueeze(0)])
            for scaled_features in scaled_features_tpl
        )
        steps = tuple(
            shifted_inputs_tpl[i] - scaled_features_tpl[i]
            for i in range(len(shifted_inputs_tpl))
        )
        scaled_grads = tuple(grads[i] * steps[i] for i in range(len(grads)))

        # aggregates across all steps for each tensor in the input tuple
        attributions = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        return attributions
