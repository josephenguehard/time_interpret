import torch

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

from torch import Tensor
from typing import Any, Callable, Tuple, Union

try:
    from transformers import PreTrainedModel
except ImportError:
    PreTrainedModel = None


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
        https://github.com/INK-USC/DIG \n
        https://arxiv.org/abs/2108.13654

    Examples:
        >>> import torch as th
        >>> from tint.attr import DiscretetizedIntegratedGradients
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(50, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = DiscretetizedIntegratedGradients(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    @log_usage()
    def attribute(
        self,
        scaled_features: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        Tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        """
        Attribute method.

        Args:
            scaled_features: (tensor, tuple):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            target (int, int, tuple, tensor, list): Output indices for
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
            additional_forward_args (Any): If the forward function
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
            n_steps: The number of steps used by the approximation
                method. Default: 50.
            return_convergence_delta: Indicates whether to return
                convergence delta or not. If `return_convergence_delta`
                is set to True convergence delta will be returned in
                a tuple following attributions.
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
        is_inputs_tuple = _is_tuple(scaled_features)
        scaled_features_tpl = _format_input(scaled_features)

        # Set requires_grad = True to inputs
        scaled_features_tpl = tuple(
            x.requires_grad_() for x in scaled_features_tpl
        )

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
