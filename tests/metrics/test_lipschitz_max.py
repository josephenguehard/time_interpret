import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from tint.metrics import lipschitz_max

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "additional_forward_args",
        "target",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), None, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, 0, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, False),
    ],
)
def test_lipschitz_max(
    forward_func,
    inputs,
    additional_forward_args,
    target,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        lipschitz = lipschitz_max(
            explanation_func=Saliency(forward_func).attribute,
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        assert tuple(lipschitz.shape) == (8,)
