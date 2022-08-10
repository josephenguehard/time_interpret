import pytest
import torch as th

from tint.models import MLP


@pytest.mark.parametrize(
    ["units", "bias", "dropout", "norm", "activations", "activations_final"],
    [
        ([10, 5, 1], True, 0.0, None, "relu", None),
        ([10, 5, 1], False, 0.0, None, "relu", None),
        ([10, 5, 1], True, 0.1, None, "relu", None),
        ([10, 5, 1], True, 0.0, "batch_norm_1d", "relu", None),
        ([10, 5, 1], True, 0.0, None, "elu", None),
        ([10, 5, 1], True, 0.0, None, "relu", "softmax"),
    ],
)
def test_init(units, bias, dropout, norm, activations, activations_final):
    mlp = MLP(
        units=units,
        bias=bias,
        dropout=dropout,
        norm=norm,
        activations=activations,
        activation_final=activations_final,
    )
    assert isinstance(mlp, MLP)


@pytest.mark.parametrize(
    [
        "units",
        "bias",
        "dropout",
        "norm",
        "activations",
        "activations_final",
        "outputs",
    ],
    [
        ([10, 5, 1], True, 0.0, None, "relu", None, (32, 1)),
        ([10, 5, 1], False, 0.0, None, "relu", None, (32, 1)),
        ([10, 5, 1], True, 0.1, None, "relu", None, (32, 1)),
        ([10, 5, 1], True, 0.0, "batch_norm_1d", "relu", None, (32, 1)),
        ([10, 5, 1], True, 0.0, None, "elu", None, (32, 1)),
        ([10, 5, 1], True, 0.0, None, "relu", "softmax", (32, 1)),
    ],
)
def test_mlp(
    units, bias, dropout, norm, activations, activations_final, outputs
):
    mlp = MLP(
        units=units,
        bias=bias,
        dropout=dropout,
        norm=norm,
        activations=activations,
        activation_final=activations_final,
    )
    x = th.rand(32, 10)
    out = mlp.forward(x)
    assert tuple(out.shape) == outputs
