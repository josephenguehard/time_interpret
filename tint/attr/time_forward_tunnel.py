import torch
import warnings

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
from typing import Any, Tuple, Union, cast


class TimeForwardTunnel(Attribution):
    def __init__(
        self,
        attribution_method: Attribution,
        task: str = "binary",
        threshold: float = 0.5,
    ) -> None:
        r"""
        Performs interpretation method by iteratively retrieving the input data
        up to a time, and computing the predictions using this data and the
        forward_func.

        This method allows to use interpretation methods in a setting which is
        not retrospective: the true label is not yet known.

        The target will be ignored when using this method, as it will be
        computed internally.

        Args:
            attribution_method (Attribution): An instance of any attribution algorithm
                        of type `Attribution`. E.g. Integrated Gradients,
                        Conductance or Saliency.
            task (str): Type of task done by the model. Either ``'binary'``,
                        ``'multilabel'``, ``'multiclass'`` or ``'regression'``.
                        Default to ``'binary'``
            threshold (float): Threshold for the multilabel task.
                        Default to 0.5
        """
        self.attribution_method = attribution_method
        self.is_delta_supported = (
            self.attribution_method.has_convergence_delta()
        )
        self._multiply_by_inputs = self.attribution_method.multiplies_by_inputs
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )
        Attribution.__init__(self, self.attribution_method.forward_func)
        self.task = task
        self.threshold = threshold

        assert task in [
            "binary",
            "multilabel",
            "multiclass",
            "regression",
        ], "task is not recognised."
        assert 0 <= threshold <= 1, "threshold must be between 0 and 1"

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
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

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Check if needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # Pop target, this param is ignore
        target = kwargs.pop("target")
        if target is not None:
            warnings.warn(
                f"target is ignored when using {self.__class__.__name__}"
            )

        attributions_partial_list = list()
        delta_partial_list = list()
        is_attrib_tuple = True
        for time in range(inputs[0].shape[1]):
            partial_inputs = tuple(x[:, :time, ...] for x in inputs)
            partial_targets = self.get_target(partial_inputs=partial_inputs)

            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = self.compute_partial_attribution(
                partial_inputs=partial_inputs,
                partial_targets=partial_targets,
                is_inputs_tuple=is_inputs_tuple,
                return_convergence_delta=return_convergence_delta,
                kwargs_partition=kwargs,
            )
            attributions_partial_list.append(attributions_partial)
            delta_partial_list.append(delta_partial)

        attributions = tuple()
        for i in range(len(attributions_partial_list[0])):
            attributions += (
                torch.cat([x[i] for x in attributions_partial_list]),
            )

        delta = None
        if self.is_delta_supported and return_convergence_delta:
            delta = torch.cat(delta_partial_list, dim=0)

        return self._apply_checks_and_return_attributions(
            attributions,
            is_attrib_tuple,
            return_convergence_delta,
            delta,
        )

    def get_target(self, partial_inputs: Tuple[Tensor]):
        """
        Get the target given partial inputs and a task.

        Args:
            partial_inputs (tuple): The partial input up to a certain time.

        Returns:
            partial_targets (tuple): The partial targets.
        """
        partial_targets = tuple(
            self.attribution_method.forward_func(x) for x in partial_inputs
        )

        if self.task in ["binary", "multiclass"]:
            partial_targets = tuple(
                torch.argmax(x, -1) for x in partial_targets
            )
        elif self.task == "multilabel":
            partial_targets = tuple(
                (x > self.threshold).float() for x in partial_targets
            )

        return partial_targets

    def compute_partial_attribution(
        self,
        partial_inputs: Tuple[Tensor, ...],
        partial_targets: Tuple[Tensor, ...],
        is_inputs_tuple: bool,
        return_convergence_delta: bool,
        kwargs_partition: Any,
    ) -> Tuple[Tuple[Tensor, ...], bool, Union[None, Tensor]]:
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method,  # self
            partial_inputs if is_inputs_tuple else partial_inputs[0],
            target=partial_targets,
            **kwargs_partition,
        )
        delta = None

        if self.is_delta_supported and return_convergence_delta:
            attributions, delta = attributions

        is_attrib_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)

        return (
            cast(Tuple[Tensor, ...], attributions),
            cast(bool, is_attrib_tuple),
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
