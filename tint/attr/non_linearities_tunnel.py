import torch.nn as nn
import torch.nn.functional as F

from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_inputs,
    _is_tuple,
    _format_tensor_into_tuples,
    _format_output,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from torch import Tensor
from typing import Any, Tuple, Union, cast


class NonLinearitiesTunnel(Attribution):
    r"""
    Replace non linearities (or any module) with others before running
    an attribution method. This tunnel is originally intended to
    replace ReLU activations with Softplus to smooth the explanations.

    .. warning::
        This method will break if the forward_func contains functional
        non linearities with additional arguments that need to be replaced.
        For instance, replacing ``F.softmax(x, dim=-1)`` is not possible due
        to the presence of the extra argument ``dim``.

    .. hint::
        In order to replace any layer, a nn.Module must be passed as
        forward_func. In particular, passing ``model.forward`` will result
        in not replacing any layer in ``model``.

    Args:
        attribution_method (Attribution): An instance of any attribution
            algorithm of type `Attribution`. E.g. Integrated Gradients,
            Conductance or Saliency.

    References:
        #. `Time Interpret: a Unified Model Interpretability Library for Time Series <https://arxiv.org/abs/2306.02968>`_
        #. `Explanations can be manipulated and geometry is to blame <https://arxiv.org/abs/1906.07983>`_

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
        replace_with: Union[nn.Module, Tuple[nn.Module, ...]] = nn.Softplus(),
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
            to_replace (nn.Module, tuple, optional): Non linearities
                to be  replaced. The linearities of type listed here will be
                replaced by ``replaced_by`` non linearities before running
                the attribution method. This can be an instance or a class.
                If a class is passed, default attributes are used.
                Default: nn.ReLU()
            replace_with (nn.Module, tuple, optional): Non linearities
                to replace the ones listed in ``to_replace``.
                Default: nn.Softplus()
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

        inputs = _format_inputs(inputs)

        # Check if needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        _replaced_layers_tpl = None
        _replaced_functions = dict()
        try:
            # Replace layers using to_replace and replace_with
            if not isinstance(to_replace, tuple):
                to_replace = (to_replace,)
            if not isinstance(replace_with, tuple):
                replace_with = (replace_with,)

            if isinstance(self.attribution_method.forward_func, nn.Module):
                _replaced_layers_tpl = tuple(
                    replace_layers(
                        self.attribution_method.forward_func, old, new
                    )
                    for old, new in zip(to_replace, replace_with)
                )

            # Replace functional using to_replace and replace_with
            for old, new in zip(to_replace, replace_with):
                name, _ = get_functional(old)
                _, func = get_functional(new)
                _replaced_functions[name] = getattr(F, name)
                setattr(F, name, func)

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

        # Even if any error is raised, restore layers and functions
        # before raising
        finally:
            # Restore layers
            if _replaced_layers_tpl is not None:
                for layer in _replaced_layers_tpl:
                    reverse_replace_layers(
                        self.attribution_method.forward_func,
                        layer,
                    )

            # Restore functions
            for name, func in _replaced_functions.items():
                setattr(F, name, func)

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
    replaced_layers = dict()
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replaced_layers[n] = replace_layers(module, old, new)

        if isinstance(module, old if type(old) == type else type(old)):
            replaced_layers[n] = getattr(model, n)
            setattr(model, n, new)
    return replaced_layers


def reverse_replace_layers(model, replaced_layers):
    """
    Reverse the layer replacement using the ``replaced_layers``
    dictionary created when running ``replace_layers``.
    """
    for k, v in replaced_layers.items():
        if isinstance(v, dict):
            reverse_replace_layers(getattr(model, k), v)
        else:
            setattr(model, k, v)


def get_functional(module):
    """
    Map a nn Non linearity to a corresponding function.
    It returns the name of the function to be replaced and the function
    to replace it with.
    """
    # Get default instance if ony type provided
    if type(module) == type:
        module = module()

    assert isinstance(module, nn.Module), "You must provide a PyTorch Module."

    if isinstance(module, nn.Threshold):
        threshold = module.threshold
        value = module.value
        inplace = module.inplace
        if inplace:
            return "threshold_", lambda x: F.threshold_(x, threshold, value)
        return "threshold", lambda x: F.threshold(x, threshold, value)

    if isinstance(module, nn.ReLU):
        inplace = module.inplace
        if inplace:
            return "relu_", F.relu_
        return "relu", F.relu

    if isinstance(module, nn.ReLU6):
        inplace = module.inplace
        return "relu6", lambda x: F.relu6(x, inplace)

    if isinstance(module, nn.ELU):
        alpha = module.alpha
        inplace = module.inplace
        if inplace:
            return "elu_", lambda x: F.elu_(x, alpha)
        return "elu", lambda x: F.elu(x, alpha)

    if isinstance(module, nn.CELU):
        alpha = module.alpha
        inplace = module.inplace
        if inplace:
            return "celu_", lambda x: F.celu_(x, alpha)
        return "celu", lambda x: F.celu(x, alpha)

    if isinstance(module, nn.LeakyReLU):
        negative_slope = module.negative_slope
        inplace = module.inplace
        if inplace:
            return "leaky_relu_", lambda x: F.leaky_relu_(x, negative_slope)
        return "leaky_relu", lambda x: F.leaky_relu(x, negative_slope)

    if isinstance(module, nn.Softplus):
        beta = module.beta
        threshold = module.threshold
        return "softplus", lambda x: F.softplus(x, beta, threshold)

    if isinstance(module, nn.Softmax):
        dim = module.dim
        return "softmax", lambda x: F.softmax(x, dim=dim)

    if isinstance(module, nn.LogSoftmax):
        dim = module.dim
        return "log_softmax", lambda x: F.log_softmax(x, dim=dim)

    if isinstance(module, nn.Sigmoid):
        return "sigmoid", F.sigmoid

    if isinstance(module, nn.Tanh):
        return "tanh", F.tanh

    if isinstance(module, nn.Tanhshrink):
        return "tanhshrink", F.tanhshrink

    return None
