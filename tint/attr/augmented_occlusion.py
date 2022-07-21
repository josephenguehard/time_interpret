from captum.attr import Occlusion
from captum.log import log_usage
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from typing import Any, Callable, Tuple, Union


class AugmentedOcclusion(Occlusion):
    def __init__(self, forward_func: Callable):
        super().__init__(forward_func=forward_func)

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
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        pass
