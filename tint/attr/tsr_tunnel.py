import numpy as np
import torch

from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from captum.log import log_usage
from captum._utils.common import _format_output
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Tuple, Union

from .occlusion import FeatureAblation, Occlusion


class TSRTunnel(Occlusion):
    r"""
    Two-step temporal saliency rescaling Tunnel.

    Performs a two-step interpretation method:
    - Mask all features at each time and compute the difference in the
      resulting attribution.
    - Mask each feature at each time and compute the difference in the
      resulting attribution, if the result of the first step is higher
      than a threshold.

    By default, the masked features are replaced with zeros. However, a
    custom baseline can also be passed.

    Using the arguments ``sliding_window_shapes`` and ``strides``, different
    alternatives of TSR can be used:
    - If sliding_window_shapes = (1, 1, ...) and strides = 1, the
      Feature-Relevance Score is computed by masking each feature individually
      providing the Time-Relevance Score is above the threshold. This
      corresponds to the Temporal Saliency Rescaling method (Algorithm 1).
    - If sliding_window_shapes = (G, G, ...) and strides = G, the
      Feature-Relevance Score is computed by masking each feature as a group
      of G features. This corresponds to the Temporal Saliency Rescaling (TSR)
      With Feature Grouping method (Algorithm 2).
    - If sliding_window_shapes = inputs.shape[2:] and strides = 1, the
      Feature-Relevance Score is computed

    Args:
        attribution_method (Attribution): An instance of any attribution algorithm
                    of type `Attribution`. E.g. Integrated Gradients,
                    Conductance or Saliency.

    References:
        `Benchmarking Deep Learning Interpretability in Time Series Predictions <https://arxiv.org/abs/2010.13924>`_

    Examples:
        >>> import torch as th
        >>> from captum.attr import Saliency
        >>> from tint.attr import TSRTunnel
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = TSRTunnel(Saliency(mlp))
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
        Occlusion.__init__(self, self.attribution_method.forward_func)
        self.use_weights = False  # We do not use weights for this method

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        threshold: float = 0.5,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
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
            sliding_window_shapes (tuple or tuple of tuples): Shape of patch
                (hyperrectangle) to occlude each input. For a single
                input tensor, this must be a tuple of length equal to the
                number of dimensions of the input tensor - 1, defining
                the dimensions of the patch. If the input tensor is 1-d,
                this should be an empty tuple. For multiple input tensors,
                this must be a tuple containing one tuple for each input
                tensor defining the dimensions of the patch for that
                input tensor, as described for the single tensor case.
            strides (int or tuple or tuple of ints or tuple of tuples, optional):
                This defines the step by which the occlusion hyperrectangle
                should be shifted by in each direction for each iteration.
                For a single tensor input, this can be either a single
                integer, which is used as the step size in each direction,
                or a tuple of integers matching the number of dimensions
                in the occlusion shape, defining the step size in the
                corresponding dimension. For multiple tensor inputs, this
                can be either a tuple of integers, one for each input
                tensor (used for all dimensions of the corresponding
                tensor), or a tuple of tuples, providing the stride per
                dimension for each tensor.
                To ensure that all inputs are covered by at least one
                sliding window, the stride for any dimension must be
                <= the corresponding sliding window dimension if the
                sliding window dimension is less than the input
                dimension.
                If None is provided, a stride of 1 is used for each
                dimension of each input tensor.
                Default: None
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
            target (int, tuple, Tensor, or list, optional): Output indices for
                which difference is computed (for classification cases,
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
                correspond to the number of examples. For all other types,
                the given argument is used for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            threshold (float): Threshold for the second step computation.
                Default: 0.5
            perturbations_per_eval (int, optional): Allows multiple occlusions
                to be included in one batch (one call to forward_fn).
                By default, perturbations_per_eval is 1, so each occlusion
                is processed individually.
                Each forward pass will contain a maximum of
                perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the
                available devices, so evaluations on each available
                device contain at most
                (perturbations_per_eval * #examples) / num_devices
                samples.
                Default: 1
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
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                The attributions with respect to each input feature.
                Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Compute sliding window for the Time-Relevance Score
        # Only the time dimension (dim 1) has a sliding window of 1
        tsr_sliding_window_shapes = tuple(
            (1,) + input.shape[2:] for input in inputs
        )

        # Compute the Time-Relevance Score (step 1)
        time_relevance_score = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
            kwargs_run_forward=kwargs,
        )

        # Reshape the Time-Relevance Score and sum along the diagonal
        assert all(
            len(tsr) == input.numel()
            for input, tsr in zip(inputs, time_relevance_score)
        ), "The attribution method must return a tensor of the same shape as the inputs"
        time_relevance_score = tuple(
            tsr.reshape(input.shape + input.shape[1:])
            for input, tsr in zip(inputs, time_relevance_score)
        )
        dim_to_sum = tuple(
            tuple(range(2, len(input.shape)))
            + tuple(range(len(input.shape) + 1, 2 * len(input.shape) - 1))
            for input in inputs
        )
        time_relevance_score = tuple(
            tsr.sum(dim) for tsr, dim in zip(time_relevance_score, dim_to_sum)
        )
        time_relevance_score = tuple(
            torch.diagonal(tsr, dim1=1, dim2=2) for tsr in time_relevance_score
        )

        # Get indexes where the Time-Relevance Score is
        # higher than the threshold
        is_above_threshold = tuple(
            score > threshold for score in time_relevance_score
        )

        # Formatting strides
        # Set temporal dim to 1
        strides = _format_and_verify_strides(strides, inputs)
        strides = tuple((1,) + strides[1:])

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            (1,) + sliding_window_shapes, inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct number of steps taking the threshold into account
        shift_counts = []
        for i, inp in enumerate(inputs):
            current_shape = np.subtract(
                inp.shape[2:], sliding_window_shapes[i][1:]
            )
            shift_counts.append(
                tuple(
                    np.add(
                        np.ceil(np.divide(current_shape, strides[i])).astype(
                            int
                        ),
                        1,
                    )
                )
            )

        # Compute Feature-Relevance Score (step 2)
        features_relevance_score = FeatureAblation.attribute.__wrapped__(
            self,
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            sliding_window_tensors=sliding_window_tensors,
            shift_counts=tuple(shift_counts),
            is_above_threshold=is_above_threshold,
            strides=strides,
            attributions_fn=abs,
            show_progress=show_progress,
            kwargs_run_forward=kwargs,
        )

        # Reshape the Feature-Relevance Score and sum along the diagonal
        features_relevance_score = tuple(
            fsr.reshape(input.shape + input.shape[1:])
            for input, fsr in zip(inputs, features_relevance_score)
        )
        dim_to_sum = tuple((1,) + (len(input.shape),) for input in inputs)
        features_relevance_score = tuple(
            fsr.sum(dim)
            for fsr, dim in zip(features_relevance_score, dim_to_sum)
        )
        diag_elts = list()
        for i, inp in enumerate(inputs):
            score = features_relevance_score[i]
            for j in range(len(inp.shape[2:]) + 1, 1, -1):
                score = torch.diagonal(score, dim1=1, dim2=j)
                diag_elts.append(score)
        features_relevance_score = tuple(diag_elts)

        # Reshape attributions before merge
        time_relevance_score = tuple(
            tsr.reshape(input.shape[:2] + (1,) * len(input.shape[2:]))
            for input, tsr in zip(inputs, time_relevance_score)
        )
        features_relevance_score = tuple(
            fsr.reshape((input.shape[0], 1) + input.shape[2:])
            for input, fsr in zip(inputs, features_relevance_score)
        )
        is_above_threshold = tuple(
            is_above.reshape(input.shape[:2] + (1,) * len(input.shape[2:]))
            for input, is_above in zip(inputs, is_above_threshold)
        )

        # Merge attributions:
        # Time-Relevance Score x Feature-Relevance Score x is above threshold
        attributions = tuple(
            tsr * frs * is_above.float()
            for tsr, frs, is_above in zip(
                time_relevance_score,
                features_relevance_score,
                is_above_threshold,
            )
        )

        return _format_output(is_inputs_tuple, attributions)

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.

        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                    kwargs.get("is_above_threshold", None),
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))

        if kwargs.get("is_above_threshold", None) is None:
            return ablated_tensor, input_mask
        return ablated_tensor, input_mask.reshape(
            (1, -1) + (1,) * len(expanded_input.shape[2:])
        )

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: Union[int, Tuple[int, ...]],
        shift_counts: Tuple[int, ...],
        is_above_threshold: Tensor = None,
    ) -> Tensor:
        """
        This constructs the current occlusion mask, which is the appropriate
        shift of the sliding window tensor based on the ablated feature number.
        The feature number ranges between 0 and the product of the shift counts
        (# of times the sliding window should be shifted in each dimension).

        First, the ablated feature number is converted to the number of steps in
        each dimension from the origin, based on shift counts. This procedure
        is similar to a base conversion, with the position values equal to shift
        counts. The feature number is first taken modulo shift_counts[0] to
        get the number of shifts in the first dimension (each shift
        by shift_count[0]), and then divided by shift_count[0].
        The procedure is then continued for each element of shift_count. This
        computes the total shift in each direction for the sliding window.

        We then need to compute the padding required after the window in each
        dimension, which is equal to the total input dimension minus the sliding
        window dimension minus the (left) shift amount. We construct the
        array pad_values which contains the left and right pad values for each
        dimension, in reverse order of dimensions, starting from the last one.

        Once these padding values are computed, we pad the sliding window tensor
        of 1s with 0s appropriately, which is the corresponding mask,
        and the result will match the input shape.
        """
        if is_above_threshold is None:
            return super()._occlusion_mask(
                expanded_input=expanded_input,
                ablated_feature_num=ablated_feature_num,
                sliding_window_tsr=sliding_window_tsr,
                strides=strides,
                shift_counts=shift_counts,
            )

        bsz = expanded_input.shape[1]
        is_above_idx = is_above_threshold.nonzero()
        shift_count_t = (
            torch.unique(is_above_idx[:, 0], return_counts=True)[1]
            .max()
            .item()
        )

        padded_tensor = torch.zeros(expanded_input.shape[1:])
        for batch_idx in range(bsz):
            remaining_total = ablated_feature_num
            if (
                len(is_above_idx[is_above_idx[:, 0] == batch_idx])
                > remaining_total % shift_count_t
            ):
                current_index = [
                    is_above_idx[is_above_idx[:, 0] == batch_idx][
                        remaining_total % shift_count_t
                    ][1].item(),
                ]
                remaining_total = remaining_total // shift_count_t
                for i, shift_count in enumerate(shift_counts):
                    stride = (
                        strides[i] if isinstance(strides, tuple) else strides
                    )
                    current_index.append(
                        (remaining_total % shift_count) * stride
                    )
                    remaining_total = remaining_total // shift_count

                remaining_padding = np.subtract(
                    expanded_input.shape[2:],
                    np.add(current_index, sliding_window_tsr.shape),
                )
                pad_values = [
                    val
                    for pair in zip(remaining_padding, current_index)
                    for val in pair
                ]
                pad_values.reverse()
                padded_tensor[batch_idx] = torch.nn.functional.pad(
                    sliding_window_tsr.unsqueeze(0), tuple(pad_values)  # type: ignore
                )

        return padded_tensor

    def _get_feature_range_and_mask(
        self, input: Tensor, input_mask: Tensor, **kwargs: Any
    ) -> Tuple[int, int, None]:
        feature_max = np.prod(kwargs["shift_counts"])
        if "is_above_threshold" in kwargs:
            feature_max *= (
                torch.unique(
                    kwargs["is_above_threshold"].nonzero()[:, 0],
                    return_counts=True,
                )[1]
                .max()
                .item()
            )
        return 0, feature_max, None

    def _get_feature_counts(self, inputs, feature_mask, **kwargs):
        """return the numbers of possible input features"""
        feature_counts = tuple(
            np.prod(counts).astype(int) for counts in kwargs["shift_counts"]
        )
        if "is_above_threshold" in kwargs:
            feature_counts = tuple(
                counts
                * torch.unique(is_above.nonzero()[:, 0], return_counts=True)[1]
                .max()
                .item()
                for input, counts, is_above in zip(
                    inputs, feature_counts, kwargs["is_above_threshold"]
                )
            )
        return feature_counts

    def _run_forward(self, *args, **kwargs) -> Tensor:
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method, *args[1:], **kwargs
        )

        # Check if it needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # If the method returns delta, we ignore it
        if self.is_delta_supported and return_convergence_delta:
            attributions, _ = attributions

        # If tuple returned, take the first element
        if isinstance(attributions, tuple):
            attributions = attributions[0]

        return attributions
