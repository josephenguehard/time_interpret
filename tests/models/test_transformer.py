import pytest
import torch as th

from tint.models import TransformerEncoder


@pytest.mark.parametrize(
    [
        "d_model",
        "nhead",
        "dim_feedforward",
        "num_layers",
        "dropout",
        "activation",
        "layer_norm_eps",
        "norm_first",
        "enable_nested_tensor",
        "many_to_one",
    ],
    [
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, False, False),
        (10, 2, 32, 1, 0.0, "relu", 1e-5, False, False, False),
        (10, 1, 32, 2, 0.0, "relu", 1e-5, False, False, False),
        (10, 1, 32, 1, 0.1, "relu", 1e-5, False, False, False),
        (10, 1, 32, 1, 0.0, "elu", 1e-5, False, False, False),
        (10, 1, 32, 1, 0.0, "relu", 1e-3, False, False, False),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, True, False, False),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, True, False),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, False, True),
    ],
)
def test_init(
    d_model,
    nhead,
    dim_feedforward,
    num_layers,
    dropout,
    activation,
    layer_norm_eps,
    norm_first,
    enable_nested_tensor,
    many_to_one,
):
    transformer = TransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        enable_nested_tensor=enable_nested_tensor,
        many_to_one=many_to_one,
    )
    assert isinstance(transformer, TransformerEncoder)


@pytest.mark.parametrize(
    [
        "d_model",
        "nhead",
        "dim_feedforward",
        "num_layers",
        "dropout",
        "activation",
        "layer_norm_eps",
        "norm_first",
        "enable_nested_tensor",
        "many_to_one",
        "outputs",
    ],
    [
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, False, False, (8, 5, 32)),
        (10, 2, 32, 1, 0.0, "relu", 1e-5, False, False, False, (8, 5, 32)),
        (10, 1, 32, 2, 0.0, "relu", 1e-5, False, False, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.1, "relu", 1e-5, False, False, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.0, "elu", 1e-5, False, False, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.0, "relu", 1e-3, False, False, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, True, False, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, True, False, (8, 5, 32)),
        (10, 1, 32, 1, 0.0, "relu", 1e-5, False, False, True, (8, 32)),
    ],
)
def test_transformer(
    d_model,
    nhead,
    dim_feedforward,
    num_layers,
    dropout,
    activation,
    layer_norm_eps,
    norm_first,
    enable_nested_tensor,
    many_to_one,
    outputs,
):
    transformer = TransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        enable_nested_tensor=enable_nested_tensor,
        many_to_one=many_to_one,
    )
    x = th.rand(8, 5, 10)
    out = transformer.forward(x)
    assert tuple(out.shape) == outputs
