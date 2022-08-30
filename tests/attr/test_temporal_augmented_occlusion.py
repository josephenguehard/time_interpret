import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import TemporalAugmentedOcclusion
from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "data", "n_sampling", "is_temporal", "fails"],
    [
        (BasicModel(), th.rand(8, 5, 2), 1, False, False),
        (BasicModel5_MultiArgs(), th.rand(8, 5, 2), 1, False, False),
        (BasicModel(), th.rand(8, 5, 2), -1, False, True),
        (BasicModel(), th.rand(8, 5, 2), 0, False, True),
        (BasicModel(), th.rand(8, 5, 2), 1, True, False),
    ],
)
def test_init(forward_func, data, n_sampling, is_temporal, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = TemporalAugmentedOcclusion(
            forward_func=forward_func,
            data=data,
            n_sampling=n_sampling,
            is_temporal=is_temporal,
        )
        assert isinstance(explainer, TemporalAugmentedOcclusion)


@pytest.mark.parametrize(
    [
        "forward_func",
        "data",
        "n_sampling",
        "is_temporal",
        "inputs",
        "sliding_window_shapes",
        "strides",
        "target",
        "additional_forward_args",
        "perturbations_per_eval",
        "attributions_fn",
        "show_progress",
        "fails",
    ],
    [
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, None, None, 1, None, False, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(12, 5, 3),
            1,
            True,
            th.rand(8, 5, 3),
            (1,),
            None,
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            1,
            None,
            False,
            False,
        ),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (2,), None, None, None, 1, None, False, False),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (2, 3), None, None, None, 1, None, False, True),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), (1,), None, None, 1, None, False, False),
        (
            BasicModel(),
            th.rand(12, 5, 3),
            1,
            True,
            th.rand(8, 5, 3),
            (1,),
            (1, 1),
            None,
            None,
            1,
            None,
            False,
            True,
        ),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, 0, None, 1, None, False, False),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, None, None, 2, None, False, False),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, None, None, -1, None, False, True),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, None, None, 1, abs, False, False),
        (BasicModel(), th.rand(12, 5, 3), 1, True, th.rand(8, 5, 3), (1,), None, None, None, 1, None, True, False),
    ],
)
def test_temporal_augmented_occlusion(
    forward_func,
    data,
    n_sampling,
    is_temporal,
    inputs,
    sliding_window_shapes,
    strides,
    target,
    additional_forward_args,
    perturbations_per_eval,
    attributions_fn,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = TemporalAugmentedOcclusion(
            forward_func=forward_func,
            data=data,
            n_sampling=n_sampling,
            is_temporal=is_temporal,
        )

        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=attributions_fn,
            show_progress=show_progress,
        )
        assert attr.shape == inputs.shape
