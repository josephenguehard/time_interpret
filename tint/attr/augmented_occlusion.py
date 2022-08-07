import numpy as np
import torch

from captum.attr import Occlusion
from captum.log import log_usage
from captum._utils.common import _format_input
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from tint.utils import _validate_input

from torch import Tensor
from typing import Any, Callable, Tuple, Union


class AugmentedOcclusion(Occlusion):
    """
    Augmented Occlusion by sampling the baseline from a bootstrapped
    distribution.

    Args:
        forward_func (callable): The forward function of the model or
            any modification of it
        data (tuple, Tensor): The data from which the baselines are sampled.
        n_sampling (int): Number of sampling to run for each occlusion.
            Default to 1
        is_temporal (bool): Whether the data is temporal or not.
            If ``True``, the data will be ablated to the inputs
            on the temporal dimension (dimension 1). Default to ``False``
    """

    def __init__(
        self,
        forward_func: Callable,
        data: TensorOrTupleOfTensorsGeneric,
        n_sampling: int = 1,
        is_temporal: bool = False,
    ):
        super().__init__(forward_func=forward_func)
        self.data = _format_input(data)
        self.n_sampling = n_sampling
        self.is_temporal = is_temporal

        assert (
            isinstance(n_sampling, int) and n_sampling >= 1
        ), "N sampling must be an integer and at least 1."

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """

        Args:
            inputs (tensor or tuple of tensors):  Input for which occlusion
                    attributions are computed. If forward_func takes a single
                    tensor as input, a single input tensor should be provided.
                    If forward_func takes multiple tensors as input, a tuple
                    of the input tensors should be provided. It is assumed
                    that for all given input tensors, dimension 0 corresponds
                    to the number of examples (aka batch size), and if
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
            target (int, tuple, tensor or list, optional):  Output indices for
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
            show_progress (bool, optional): Displays the progress of
                    computation. It will try to use tqdm if available for
                    advanced features (e.g. time estimation). Otherwise, it
                    will fallback to a simple output of progress.
                    Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                    The attributions with respect to each input feature.
                    Attributions will always be
                    the same size as the provided inputs, with each value
                    providing the attribution of the corresponding input index.
                    If a single tensor is provided as inputs, a single tensor
                    is returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.
        """
        # Change input to tuple and check that its length is the same as data.
        # Also check that each dimension between inputs and self.data matches
        # except on the first one.
        formatted_inputs = _format_input(inputs)
        _validate_input(
            inputs=formatted_inputs,
            data=self.data,
            is_temporal=self.is_temporal,
        )

        return super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            # Baselines are used here to keep track of the input index
            # The true baselines will be sampled from self.data
            baselines=tuple(range(len(inputs))),
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=False,
        )

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
        Ablates given expanded_input tensor with given feature mask, feature
        range, and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are
        provided in kwargs. baseline is expected to
        be broadcastable to match expanded_input.

        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated
        (and thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()

        # We repeat input_mask n_sampling times
        input_mask = torch.cat([input_mask] * self.n_sampling, dim=0)

        # We ablate data if temporal on the time dimension (dimension 1)
        data = self.data[baseline]
        if self.is_temporal:
            time_shape = expanded_input.shape[2]
            data = data[:, :time_shape, ...]

        # We replace the original baseline with samples from a bootstrapped
        # distribution over self.data.
        # We query perturbations_per_eval x len(input) samples and reshape
        # The baseline afterwards.
        # The input baseline is used to get the index of the input.
        size = expanded_input.shape[0] * expanded_input.shape[1]
        baseline = torch.index_select(
            data,
            0,
            torch.randint(high=len(data), size=(size,)),
        )
        baseline = baseline.reshape((-1,) + expanded_input.shape[1:])

        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))
        return ablated_tensor, input_mask

    def _get_feature_range_and_mask(
        self, input: Tensor, input_mask: Tensor, **kwargs: Any
    ) -> Tuple[int, int, None]:
        feature_max = int(np.prod(kwargs["shift_counts"]))
        return 0, feature_max * self.n_sampling, None
