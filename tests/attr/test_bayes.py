import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import BayesLime, BayesKernelShap
from tint.attr.models import BLRRegression, BLRRidge
from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "interpretable_model", "fails"],
    [
        (BasicModel(), None, False),
        (BasicModel(), BLRRegression(), False),
        (BasicModel(), BLRRidge(), False),
    ],
)
def test_init(forward_func, interpretable_model, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesLime(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
        )
        assert isinstance(explainer, BayesLime)

    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesKernelShap(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
        )
        assert isinstance(explainer, BayesKernelShap)


@pytest.mark.parametrize(
    [
        "forward_func",
        "interpretable_model",
        "inputs",
        "baselines",
        "target",
        "additional_forward_args",
        "feature_mask",
        "perturbations_per_eval",
        "return_input_shape",
        "show_progress",
        "fails",
    ],
    [
        (BasicModel(), None, th.rand(1, 5, 3), None, None, None, None, 1, True, False, False),
        (BasicModel(), BLRRegression(), th.rand(1, 5, 3), None, None, None, None, 1, True, False, False),
        (BasicModel(), None, th.rand(1, 5, 3), th.zeros(1, 5, 3), None, None, None, 1, True, False, False),
        (BasicModel(), None, th.rand(1, 5, 3), None, 0, None, None, 1, True, False, False),
        (
            BasicModel5_MultiArgs(),
            None,
            th.rand(1, 5, 3),
            None,
            None,
            (th.rand(1, 5, 3), th.rand(1, 5, 3)),
            None,
            1,
            True,
            False,
            False,
        ),
        (BasicModel(), None, th.rand(1, 5, 3), None, None, None, None, 2, True, False, False),
        (BasicModel(), None, th.rand(1, 5, 3), None, None, None, None, 1, False, False, False),
        (BasicModel(), None, th.rand(1, 5, 3), None, None, None, None, 1, True, True, False),
    ],
)
def test_bayes(
    forward_func,
    interpretable_model,
    inputs,
    baselines,
    target,
    additional_forward_args,
    feature_mask,
    perturbations_per_eval,
    return_input_shape,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesLime(forward_func=forward_func, interpretable_model=interpretable_model)

        attr, credible_int = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )
        if return_input_shape:
            assert attr.shape == inputs.shape
            assert credible_int.shape == inputs.shape
        else:
            assert tuple(attr.shape) == (1, 15)
            assert tuple(credible_int.shape) == (15,)

    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesKernelShap(forward_func=forward_func, interpretable_model=interpretable_model)

        attr, credible_int = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )
        if return_input_shape:
            assert attr.shape == inputs.shape
            assert credible_int.shape == inputs.shape
        else:
            assert tuple(attr.shape) == (1, 15)
            assert tuple(credible_int.shape) == (15,)
