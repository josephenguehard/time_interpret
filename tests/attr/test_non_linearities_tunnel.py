import pytest
import torch as th
import torch.nn as nn

from captum.attr import Saliency

from contextlib import nullcontext

from tint.attr import NonLinearitiesTunnel

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
        explainer = NonLinearitiesTunnel(
            Saliency(
                forward_func=forward_func,
            )
        )
        assert isinstance(explainer, NonLinearitiesTunnel)


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "target",
        "additional_forward_args",
        "to_replace",
        "replace_with",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, nn.ReLU(), nn.Softplus(), False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            nn.ReLU(),
            nn.Softplus(),
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), None, None, nn.ReLU, nn.Softplus(), False),
        (BasicModel(), th.rand(8, 5, 3), None, None, (nn.ReLU(),), nn.Softplus(), False),
        (BasicModel(), th.rand(8, 5, 3), None, None, (nn.ReLU(),), (nn.Softplus(),), False),
        (BasicModel(), th.rand(8, 5, 3), None, None, (nn.ReLU(), nn.ReLU6()), (nn.Softplus(), nn.Softmax()), False),
    ],
)
def test_time_forward_tunnel(
    forward_func,
    inputs,
    target,
    additional_forward_args,
    to_replace,
    replace_with,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = NonLinearitiesTunnel(Saliency(forward_func=forward_func))

        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            to_replace=to_replace,
            replace_with=replace_with,
        )
        assert attr.shape == inputs.shape
