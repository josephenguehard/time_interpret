import torch

from captum.attr import Occlusion
from captum.log import log_usage
from captum._utils.common import _format_input, _validate_input
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple, Union


class AugmentedOcclusion(Occlusion):
    def __init__(
        self,
        forward_func: Callable,
        data: TensorOrTupleOfTensorsGeneric,
    ):
        super().__init__(forward_func=forward_func)
        self.data = _format_input(data)

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
        inputs = _format_input(inputs)
        _validate_input(
            inputs=inputs,
            baselines=self.data,
            draw_baseline_from_distrib=True,
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
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()

        # We replace the original baseline with samples from a bootstrapped
        # distribution over self.data.
        # We query perturbations_per_eval x len(input) samples and reshape
        # The baseline afterwards.
        # The input baseline is used to get the index of the input.
        size = expanded_input.shape[0] * expanded_input.shape[1]
        baseline = torch.index_select(
            self.data[baseline],
            0,
            torch.randint(high=len(self.data), size=(size,)),
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
