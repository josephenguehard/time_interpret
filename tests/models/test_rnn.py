import pytest
import torch as th

from tint.models import RNN


@pytest.mark.parametrize(
    [
        "input_size",
        "rnn",
        "hidden_size",
        "num_layers",
        "bias",
        "dropout",
        "bidirectional",
        "many_to_one",
    ],
    [
        (10, "rnn", 32, 1, True, 0.0, False, False),
        (10, "lstm", 32, 1, True, 0.0, False, False),
        (10, "rnn", 32, 2, True, 0.0, False, False),
        (10, "rnn", 32, 1, False, 0.0, False, False),
        (10, "rnn", 32, 1, True, 0.1, False, False),
        (10, "rnn", 32, 1, True, 0.0, True, False),
        (10, "rnn", 32, 1, True, 0.0, False, True),
    ],
)
def test_init(
    input_size,
    rnn,
    hidden_size,
    num_layers,
    bias,
    dropout,
    bidirectional,
    many_to_one,
):
    rnn = RNN(
        input_size=input_size,
        rnn=rnn,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        dropout=dropout,
        bidirectional=bidirectional,
        many_to_one=many_to_one,
    )
    assert isinstance(rnn, RNN)


@pytest.mark.parametrize(
    [
        "input_size",
        "rnn",
        "hidden_size",
        "num_layers",
        "bias",
        "dropout",
        "bidirectional",
        "many_to_one",
        "outputs",
    ],
    [
        (10, "rnn", 32, 1, True, 0.0, False, False, (8, 5, 32)),
        (10, "lstm", 32, 1, True, 0.0, False, False, (8, 5, 32)),
        (10, "rnn", 32, 2, True, 0.0, False, False, (8, 5, 32)),
        (10, "rnn", 32, 1, False, 0.0, False, False, (8, 5, 32)),
        (10, "rnn", 32, 1, True, 0.1, False, False, (8, 5, 32)),
        (10, "rnn", 32, 1, True, 0.0, True, False, (8, 5, 64)),
        (10, "rnn", 32, 1, True, 0.0, False, True, (8, 32)),
    ],
)
def test_rnn(
    input_size,
    rnn,
    hidden_size,
    num_layers,
    bias,
    dropout,
    bidirectional,
    many_to_one,
    outputs,
):
    rnn = RNN(
        input_size=input_size,
        rnn=rnn,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        dropout=dropout,
        bidirectional=bidirectional,
        many_to_one=many_to_one,
    )
    x = th.rand(8, 5, 10)
    out = rnn.forward(x)
    assert tuple(out.shape) == outputs
