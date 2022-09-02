import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from tint.attr import TimeForwardTunnel

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
        explainer = TimeForwardTunnel(
            Saliency(
                forward_func=forward_func,
            )
        )
        assert isinstance(explainer, TimeForwardTunnel)


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "target",
        "additional_forward_args",
        "task",
        "threshold",
        "temporal_target",
        "temporal_additional_forward_args",
        "return_temporal_attributions",
        "show_progress",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, "none", 0.5, False, None, False, False, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            "none",
            0.5,
            False,
            (True, True),
            False,
            False,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), None, None, "binary", 0.3, False, None, False, False, False),
        # (BasicModel(), th.rand(8, 5, 3), None, None, "regression", 0.5, False, None, False, False, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, "wrong", 0.5, False, None, False, False, True),
        (BasicModel(), th.rand(8, 5, 3), None, None, "none", 0.5, False, None, False, False, False),
        (BasicModel(), th.rand(8, 5, 3), 0, None, "none", 0.5, False, None, False, False, False),
        (BasicModel(), th.rand(8, 5, 3), th.randint(2, (8, 5)), None, "binary", 0.5, True, None, False, False, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, "none", 0.5, False, None, True, False, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, "none", 0.5, False, None, False, True, False),
    ],
)
def test_time_forward_tunnel(
    forward_func,
    inputs,
    target,
    additional_forward_args,
    task,
    threshold,
    temporal_target,
    temporal_additional_forward_args,
    return_temporal_attributions,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = TimeForwardTunnel(Saliency(forward_func=forward_func))

        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            task=task,
            threshold=threshold,
            temporal_target=temporal_target,
            temporal_additional_forward_args=temporal_additional_forward_args,
            return_temporal_attributions=return_temporal_attributions,
            show_progress=show_progress,
        )

        if return_temporal_attributions:
            assert tuple(attr.shape) == (inputs.shape[0], inputs.shape[1]) + inputs.shape[1:]
        else:
            assert attr.shape == inputs.shape
