import torch

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

from tint.utils import get_progress_bars, _slice_to_time


class TimeForwardTunnel(Attribution):
    def __init__(
        self,
        attribution_method: Attribution,
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

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        task: str = "none",
        threshold: float = 0.5,
        temporal_target: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        show_progress: bool = False,
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
            task (str): Type of task done by the model. Either ``'binary'``,
                        ``'multilabel'``, ``'multiclass'`` or ``'regression'``.
                        Default to ``'binary'``
            threshold (float): Threshold for the multilabel task.
                        Default to 0.5
            temporal_target (bool, optional): Determine if the targe is
                    temporal and needs to be cut.
                    Default: False
            temporal_additional_forward_args (tuple, optional): For each
                    additional forward arg, determine if it is temporal
                    or not.
                    Default: None
            return_temporal_attributions (bool): Whether to return all saliencies
                for all time points or only the last one per time point.
                    Default: False
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
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

        attributions_partial_list = list()
        delta_partial_list = list()
        is_attrib_tuple = True

        times = range(1, inputs[0].shape[1] + 1)
        if show_progress:
            times = get_progress_bars()(
                times, desc=f"{self.attribution_method.get_name()} attribution"
            )

        # Compute attributions over time
        for time in times:
            partial_inputs, kwargs_copy = _slice_to_time(
                inputs=inputs,
                time=time,
                forward_func=self.attribution_method.forward_func,
                task=task,
                threshold=threshold,
                temporal_target=temporal_target,
                temporal_additional_forward_args=temporal_additional_forward_args,
                **kwargs,
            )

            # Get partial targets
            partial_targets = kwargs_copy.pop("target", None)
            if not isinstance(partial_targets, tuple):
                partial_targets = (partial_targets,)

            # Compute attribution for a specific time
            # and for each partial target
            attributions_partial_sublist = list()
            delta_partial_list_sublist = list()
            for partial_target in partial_targets:
                (
                    attributions_partial,
                    is_attrib_tuple,
                    delta_partial,
                ) = self.compute_partial_attribution(
                    partial_inputs=partial_inputs,
                    partial_target=partial_target,
                    is_inputs_tuple=is_inputs_tuple,
                    return_convergence_delta=return_convergence_delta,
                    kwargs_partition=kwargs_copy,
                )
                attributions_partial_sublist.append(attributions_partial)
                delta_partial_list_sublist.append(delta_partial)

            # Group attributions
            attributions_partial = tuple()
            for i in range(len(attributions_partial_sublist[0])):
                attributions_partial += (
                    torch.stack(
                        [x[i] for x in attributions_partial_sublist],
                        dim=-1,
                    )
                    .max(-1)
                    .values,
                )

            # Group delta is required
            delta_partial = None
            if self.is_delta_supported and return_convergence_delta:
                delta_partial = torch.stack(
                    delta_partial_list_sublist, dim=-1
                ).mean(-1)

            attributions_partial_list.append(attributions_partial)
            delta_partial_list.append(delta_partial)

        # If return all saliencies, stack attributions
        # else, select the last one in time for each time point
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
        if self.is_delta_supported and return_convergence_delta:
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
        partial_target: Tensor,
        is_inputs_tuple: bool,
        return_convergence_delta: bool,
        kwargs_partition: Any,
    ) -> Tuple[Tuple[Tensor, ...], bool, Union[None, Tensor]]:
        if partial_target is None:
            attributions = self.attribution_method.attribute.__wrapped__(
                self.attribution_method,  # self
                partial_inputs if is_inputs_tuple else partial_inputs[0],
                **kwargs_partition,
            )
        else:
            attributions = self.attribution_method.attribute.__wrapped__(
                self.attribution_method,  # self
                partial_inputs if is_inputs_tuple else partial_inputs[0],
                target=partial_target,
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
