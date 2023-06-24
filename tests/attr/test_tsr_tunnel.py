import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from tint.attr import TSRTunnel

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "fails"],
    [
        (BasicModel(), False),
        (BasicModel5_MultiArgs(), False),
    ],
)
def test_init(forward_func, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = TSRTunnel(
            Saliency(
                forward_func=forward_func,
            )
        )
        assert isinstance(explainer, TSRTunnel)


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "sliding_window_shapes",
        "strides",
        "baselines",
        "target",
        "additional_forward_args",
        "threshold",
        "normalize",
        "perturbations_per_eval",
        "show_progress",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.0, True, 1, False, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            (1, 1),
            None,
            None,
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            0.0,
            True,
            1,
            False,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), (1,), None, None, None, None, 0.0, True, 1, False, True),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), 2, None, None, None, 0.0, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), 4, None, None, None, 0.0, True, 1, False, True),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, 0, None, None, 0.0, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, th.rand(8, 5, 3), None, None, 0.0, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, 0, None, 0.0, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, th.zeros((8,)).long(), None, 0.0, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.5, True, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.0, False, 1, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.0, True, 2, False, False),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.0, True, 0, False, True),
        (BasicModel(), th.rand(8, 5, 3), (1, 1), None, None, None, None, 0.0, True, 1, True, False),
    ],
)
def test_time_forward_tunnel(
    forward_func,
    inputs,
    sliding_window_shapes,
    strides,
    baselines,
    target,
    additional_forward_args,
    threshold,
    normalize,
    perturbations_per_eval,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = TSRTunnel(Saliency(forward_func=forward_func))

        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            threshold=threshold,
            normalize=normalize,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )

        assert attr.shape == inputs.shape
