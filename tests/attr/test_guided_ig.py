import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import GuidedIntegratedGradients

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "multiply_by_inputs", "fails"],
    [
        (BasicModel(), True, False),
        (BasicModel5_MultiArgs(), True, False),
        (BasicModel(), False, False),
    ],
)
def test_init(forward_func, multiply_by_inputs, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = GuidedIntegratedGradients(forward_func=forward_func, multiply_by_inputs=multiply_by_inputs)
        assert isinstance(explainer, GuidedIntegratedGradients)


@pytest.mark.parametrize(
    [
        "forward_func",
        "multiply_by_inputs",
        "inputs",
        "baselines",
        "target",
        "additional_forward_args",
        "n_anchors",
        "n_guided_steps",
        "fraction",
        "internal_batch_size",
        "return_convergence_delta",
        "fails",
    ],
    [
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel5_MultiArgs(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            0,
            None,
            None,
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            th.rand(8, 5, 3),
            None,
            None,
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            0,
            None,
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            -1,
            5,
            0.1,
            None,
            False,
            True,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            1,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            5,
            0.9,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            5,
            0.1,
            None,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            5,
            0.1,
            8,
            False,
            False,
        ),
        (
            BasicModel(),
            True,
            th.rand(8, 5, 3),
            None,
            None,
            None,
            10,
            5,
            0.1,
            None,
            True,
            False,
        ),
    ],
)
def test_time_forward_tunnel(
    forward_func,
    multiply_by_inputs,
    inputs,
    baselines,
    target,
    additional_forward_args,
    n_anchors,
    n_guided_steps,
    fraction,
    internal_batch_size,
    return_convergence_delta,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = GuidedIntegratedGradients(forward_func=forward_func, multiply_by_inputs=multiply_by_inputs)

        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            n_anchors=n_anchors,
            n_guided_steps=n_guided_steps,
            fraction=fraction,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=return_convergence_delta,
        )

        if return_convergence_delta:
            attr, delta = attr
            assert tuple(delta.shape) == (8,)

        assert attr.shape == inputs.shape
