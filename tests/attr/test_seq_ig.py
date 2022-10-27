import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import SequentialIntegratedGradients

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
        explainer = SequentialIntegratedGradients(forward_func=forward_func, multiply_by_inputs=multiply_by_inputs)
        assert isinstance(explainer, SequentialIntegratedGradients)


@pytest.mark.parametrize(
    [
        "forward_func",
        "multiply_by_inputs",
        "inputs",
        "baselines",
        "target",
        "additional_forward_args",
        "n_steps",
        "method",
        "internal_batch_size",
        "return_convergence_delta",
        "show_progress",
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
            "gausslegendre",
            None,
            False,
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
            "gausslegendre",
            None,
            False,
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
            "gausslegendre",
            None,
            False,
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
            "gausslegendre",
            None,
            False,
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
            "gausslegendre",
            None,
            False,
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
            "gausslegendre",
            None,
            False,
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
            "riemann_trapezoid",
            None,
            False,
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
            "gausslegendre",
            8,
            False,
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
            "gausslegendre",
            None,
            True,
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
            "gausslegendre",
            None,
            False,
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
    n_steps,
    method,
    internal_batch_size,
    return_convergence_delta,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = SequentialIntegratedGradients(forward_func=forward_func, multiply_by_inputs=multiply_by_inputs)

        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=return_convergence_delta,
            show_progress=show_progress,
        )

        if return_convergence_delta:
            attr, delta = attr
            assert tuple(delta.shape) == (8,)

        assert attr.shape == inputs.shape
