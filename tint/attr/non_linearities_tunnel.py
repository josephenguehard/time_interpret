import torch.nn as nn

from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _is_tuple,
    _format_tensor_into_tuples,
    _format_output,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from torch import Tensor
from typing import Any, Dict, Tuple, Union, cast


class NonLinearitiesTunnel(Attribution):
    r"""
    Replace non linearities (or any module) with other before running
    an attribution method. This tunnel is originally intended to
    replace ReLU activations with Softplus to smooth the explanations.

    Args:
        attribution_method (Attribution): An instance of any attribution algorithm
                    of type `Attribution`. E.g. Integrated Gradients,
                    Conductance or Saliency.

    References:
        https://arxiv.org/abs/1906.07983

    Examples:
        >>> import torch as th
        >>> from captum.attr import Saliency
        >>> from tint.attr import NonLinearitiesTunnel
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = NonLinearitiesTunnel(Saliency(mlp))
        >>> attr = explainer.attribute(inputs, target=0)
    """

    def __init__(
        self,
        attribution_method: Attribution,
    ) -> None:
        self.attribution_method = attribution_method
        self.is_delta_supported = (
            self.attribution_method.has_convergence_delta()
        )
        self._multiply_by_inputs = self.attribution_method.multiplies_by_inputs
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )
        Attribution.__init__(self, self.attribution_method.forward_func)

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        to_replace: Union[nn.Module, Tuple[nn.Module, ...]] = nn.ReLU(),
        replace_by: Union[nn.Module, Tuple[nn.Module, ...]] = nn.Softplus(
            beta=10
        ),
        replace_dict: Dict[nn.Module, nn.Module] = None,
        **kwargs: Any,
    ) -> Union[
        Union[
            Tensor,
            Tuple[Tensor, Tensor],
            Tuple[Tensor, ...],
            Tuple[Tuple[Tensor, ...], Tensor],
        ]
    ]:
        r"""

        Args:
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
                It is also assumed that for all given input tensors,
                dimension 1 corresponds to the time dimension, and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.
            to_replace (nn.Module, list, optional): List of non linearities
                to be  replaced. The linearities of type listed here will be
                replaced by ``replaced_by` non linearities before running
                the attribution method.
                Default: nn.ReLU()
            replace_by (nn.Module, list, optional): List of non linearities
                to replace non linearities listed in ``to_replace``.
                Default: nn.Softplus(beta=10)
            replace_dict (dict, optional): A dictionary where each key is
                replaced by the corresponding value.
                Default: None
            **kwargs: (Any, optional): Contains a list of arguments that are
                passed  to `attribution_method` attribution algorithm.
                Any additional arguments that should be used for the
                chosen attribution method should be included here.
                For instance, such arguments include
                `additional_forward_args` and `baselines`.

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Attribution with
                respect to each input feature. attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor
                is returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*float*, returned if return_convergence_delta=True):
                Approximation error computed by the
                attribution algorithm. Not all attribution algorithms
                return delta value. It is computed only for some
                algorithms, e.g. integrated gradients.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_input(inputs)

        # Check if needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # forward_func must be a PyTorch module
        assert isinstance(
            self.attribution_method.forward_func, nn.Module
        ), "forward_func must be a PyTorch module with this method."

        # Replace layers using to_replace and replaced_by
        if not isinstance(to_replace, tuple):
            to_replace = (to_replace,)
        if not isinstance(replace_by, tuple):
            replace_by = (replace_by,)

        _replaced_layers_tpl = (
            replace_layers(self.attribution_method.forward_func, old, new)
            for old, new in zip(to_replace, replace_by)
        )

        # If replace_dict is provided
        if replace_dict is not None:
            _replaced_layers_tpl += (
                replace_layers(self.attribution_method.forward_func, old, new)
                for old, new in replace_dict.items()
            )

        # Get attributions
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method,  # self
            inputs if is_inputs_tuple else inputs[0],
            **kwargs,
        )

        # Get delta if required
        delta = None
        if self.is_delta_supported and return_convergence_delta:
            attributions, delta = attributions

        # Format attributions
        is_attrib_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)

        # Restore non linearities
        for layer in _replaced_layers_tpl:
            reverse_replace_layers(self.attribution_method.forward_func, layer)

        return self._apply_checks_and_return_attributions(
            attributions,
            is_attrib_tuple,
            return_convergence_delta,
            delta,
        )

    def _apply_checks_and_return_attributions(
        self,
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
            if self.is_delta_supported and return_convergence_delta
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


def replace_layers(model, old, new):
    """
    Replace all the layers of type old into new.

    Returns:
        dict: Dictionary of replaced layers, saved to be restored after
            running the attribution method.

    References:
        https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/12
    """
    replaced_names = dict()
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replaced_names[n] = replace_layers(module, old, new)

        if isinstance(module, old if type(old) == type else type(old)):
            replaced_names[n] = getattr(model, n)
            setattr(model, n, new)
    return replaced_names


def reverse_replace_layers(model, replaced_names):
    """
    Reverse the layer replacement using the ``replaced_names``
    dictionary created when running ``replace_layers``.
    """
    for k, v in replaced_names.items():
        if isinstance(v, dict):
            reverse_replace_layers(getattr(model, k), v)
        else:
            setattr(model, k, v)
