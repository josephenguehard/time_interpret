import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import DiscretetizedIntegratedGradients
from tests.basic_models import BasicLinearModel, BasicLinearModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "multiply_by_inputs", "fails"],
    [
        (BasicLinearModel(), True, False),
        (BasicLinearModel5_MultiArgs(), True, False),
        (BasicLinearModel(), False, False),
    ],
)
def test_init(forward_func, multiply_by_inputs, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = DiscretetizedIntegratedGradients(
            forward_func=forward_func,
            multiply_by_inputs=multiply_by_inputs,
        )
        assert isinstance(explainer, DiscretetizedIntegratedGradients)


@pytest.mark.parametrize(
    [
        "forward_func",
        "multiply_by_inputs",
        "scaled_features",
        "target",
        "additional_forward_args",
        "n_steps",
        "return_convergence_delta",
        "fails",
    ],
    [
        (BasicLinearModel(), True, th.rand(8, 5, 3), None, None, 2, False, False),
        # (BasicLinearModel5_MultiArgs(), True, th.rand(8, 5, 3), None, (th.rand(8, 5, 4),), 2, False, False),
        (BasicLinearModel(), True, th.rand(8, 5, 3), 0, None, 2, False, False),
        (BasicLinearModel(), True, th.rand(8, 5, 3), None, None, -1, False, True),
        # (BasicLinearModel(), True, th.rand(8, 5, 3), None, None, 2, True, False),
    ],
)
def test_discretised_ig(
    forward_func,
    multiply_by_inputs,
    scaled_features,
    target,
    additional_forward_args,
    n_steps,
    return_convergence_delta,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = DiscretetizedIntegratedGradients(forward_func=forward_func, multiply_by_inputs=multiply_by_inputs)

        attr = explainer.attribute(
            scaled_features=scaled_features,
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            return_convergence_delta=return_convergence_delta,
        )
        if return_convergence_delta:
            attr = attr[0]
        assert tuple(attr.shape) == (4, 5, 3)  # The shape returned by DIG changes depending on steps.
