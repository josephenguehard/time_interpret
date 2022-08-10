import pytest
import torch as th

from tint.models import MLP, Net


@pytest.mark.parametrize(
    [
        "layers",
        "loss",
        "optim",
        "lr",
        "lr_scheduler",
        "lr_scheduler_args",
        "l2",
    ],
    [
        (MLP([5, 10, 1]), "mse", "adam", 0.001, None, None, 0.0),
        ([MLP([5, 10, 1])], "mse", "adam", 0.001, None, None, 0.0),
        ([MLP([5, 10]), MLP([10, 1])], "mse", "adam", 0.001, None, None, 0.0),
        (MLP([5, 10, 1]), "cross_entropy", "adam", 0.001, None, None, 0.0),
        (MLP([5, 10, 1]), "mse", "sgd", 0.001, None, None, 0.0),
        (MLP([5, 10, 1]), "mse", "adam", 0.1, None, None, 0.0),
        (
            MLP([5, 10, 1]),
            "mse",
            "adam",
            0.001,
            "reduce_on_plateau",
            None,
            0.0,
        ),
        (
            MLP([5, 10, 1]),
            "mse",
            "adam",
            0.001,
            "reduce_on_plateau",
            {"factor": 0.5, "patience": 5},
            0.0,
        ),
        (MLP([5, 10, 1]), "mse", "adam", 0.001, None, None, 0.01),
    ],
)
def test_init(layers, loss, optim, lr, lr_scheduler, lr_scheduler_args, l2):
    net = Net(
        layers=layers,
        loss=loss,
        optim=optim,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_args=lr_scheduler_args,
        l2=l2,
    )
    assert isinstance(net, Net)


@pytest.mark.parametrize(
    [
        "layers",
        "loss",
        "optim",
        "lr",
        "lr_scheduler",
        "lr_scheduler_args",
        "l2",
        "outputs",
    ],
    [
        (MLP([5, 10, 1]), "mse", "adam", 0.001, None, None, 0.0, (32, 1)),
        ([MLP([5, 10, 1])], "mse", "adam", 0.001, None, None, 0.0, (32, 1)),
        (
            [MLP([5, 10]), MLP([10, 1])],
            "mse",
            "adam",
            0.001,
            None,
            None,
            0.0,
            (32, 1),
        ),
        (
            MLP([5, 10, 1]),
            "cross_entropy",
            "adam",
            0.001,
            None,
            None,
            0.0,
            (32, 1),
        ),
        (MLP([5, 10, 1]), "mse", "sgd", 0.001, None, None, 0.0, (32, 1)),
        (MLP([5, 10, 1]), "mse", "adam", 0.1, None, None, 0.0, (32, 1)),
        (
            MLP([5, 10, 1]),
            "mse",
            "adam",
            0.001,
            "reduce_on_plateau",
            None,
            0.0,
            (32, 1),
        ),
        (
            MLP([5, 10, 1]),
            "mse",
            "adam",
            0.001,
            "reduce_on_plateau",
            {"factor": 0.5, "patience": 5},
            0.0,
            (32, 1),
        ),
        (MLP([5, 10, 1]), "mse", "adam", 0.001, None, None, 0.01, (32, 1)),
    ],
)
def test_net(
    layers, loss, optim, lr, lr_scheduler, lr_scheduler_args, l2, outputs
):
    net = Net(
        layers=layers,
        loss=loss,
        optim=optim,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_args=lr_scheduler_args,
        l2=l2,
    )

    x = th.rand(32, 5)
    out = net(x)
    assert tuple(out.shape) == outputs

    y = th.rand(32, 1)
    batch = {"x": x, "y": y}
    loss = net.step(batch, 0, "train").item()
    assert isinstance(loss, float)

    pred = net.predict_step(batch, 0)
    assert pred.shape == outputs
