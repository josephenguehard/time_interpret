import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from tint.metrics import log_odds
from tint.metrics.weights import lime_weights, lof_weights

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "baselines",
        "additional_forward_args",
        "target",
        "topk",
        "weight_fn",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.2, None, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            0.2,
            None,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), 0, None, None, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), th.rand(8, 5, 3), None, None, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, 0, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.6, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 1.2, None, True),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.2, lime_weights(), False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.2, lime_weights("euclidean"), False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 0.2, lof_weights(th.rand(20, 5, 3), 5), False),
    ],
)
def test_log_odds(
    forward_func,
    inputs,
    baselines,
    additional_forward_args,
    target,
    topk,
    weight_fn,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Saliency(forward_func=forward_func)
        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )

        log_odds_ = log_odds(
            forward_func=forward_func,
            inputs=inputs,
            attributions=attr,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
            topk=topk,
            weight_fn=weight_fn,
        )

        assert isinstance(log_odds_, float)
