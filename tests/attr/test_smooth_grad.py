import pytest
import torch as th

from captum.attr import NoiseTunnel

from contextlib import nullcontext

from tint.attr import SmoothGrad
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
        explainer = SmoothGrad(forward_func=forward_func)
        assert isinstance(explainer, NoiseTunnel)


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "target",
        "abs",
        "additional_forward_args",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, True, None, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            False,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), 0, False, None, False),
        (BasicModel(), th.rand(8, 5, 3), 0, True, None, False),
    ],
)
def test_smooth_grad(
    forward_func,
    inputs,
    target,
    abs,
    additional_forward_args,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = SmoothGrad(forward_func=forward_func)

        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            abs=abs,
            additional_forward_args=additional_forward_args,
        )
        assert attr.shape == inputs.shape
