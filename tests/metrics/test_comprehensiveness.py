import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from tint.metrics import comprehensiveness

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "baselines",
        "additional_forward_args",
        "target",
        "topk",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.2, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            0.2,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), 0, None, None, 0.2, False),
        (BasicModel(), th.rand(8, 5, 3), th.rand(8, 5, 3), None, None, 0.2, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, 0, 0.2, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.6, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 1.2, True),
    ],
)
def test_comprehensiveness(
    forward_func,
    inputs,
    baselines,
    additional_forward_args,
    target,
    topk,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Saliency(forward_func=forward_func)
        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )

        comp = comprehensiveness(
            forward_func=forward_func,
            inputs=inputs,
            attributions=attr,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
            topk=topk,
        )

        if target == 0:
            assert tuple(comp.shape) == (8,)
        else:
            assert tuple(comp.shape) == (8, 1)
