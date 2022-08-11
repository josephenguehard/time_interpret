import pytest
import torch as th

from contextlib import nullcontext

from tint.attr import LofLime, LofKernelShap
from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "embeddings", "n_neighbors", "fails"],
    [
        (BasicModel(), th.rand(8, 5, 3), 2, False),
        (BasicModel5_MultiArgs(), th.rand(8, 5, 3), 2, False),
        (BasicModel(), th.rand(8, 5, 3), -1, True),
    ],
)
def test_init(forward_func, embeddings, n_neighbors, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = LofLime(
            forward_func=forward_func,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
        )
        assert isinstance(explainer, LofLime)

    with pytest.raises(Exception) if fails else nullcontext():
        explainer = LofKernelShap(
            forward_func=forward_func,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
        )
        assert isinstance(explainer, LofKernelShap)


@pytest.mark.parametrize(
    [
        "forward_func",
        "embeddings",
        "n_neighbors",
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
        (BasicModel(), th.rand(12, 5, 3), 2, th.rand(8, 5, 3), None, None, None, None, 1, True, False, False),
        (
            BasicModel(),
            th.rand(12, 5, 3),
            2,
            th.rand(8, 5, 3),
            th.zeros(8, 5, 3),
            None,
            None,
            None,
            1,
            True,
            False,
            False,
        ),
        (BasicModel(), th.rand(12, 5, 3), 2, th.rand(8, 5, 3), None, 0, None, None, 1, True, False, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(12, 5, 3),
            2,
            th.rand(8, 5, 3),
            None,
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            1,
            True,
            False,
            False,
        ),
        (BasicModel(), th.rand(12, 5, 3), 2, th.rand(8, 5, 3), None, None, None, None, 2, True, False, False),
        (BasicModel(), th.rand(12, 5, 3), 2, th.rand(8, 5, 3), None, None, None, None, 1, False, False, False),
        (BasicModel(), th.rand(12, 5, 3), 2, th.rand(8, 5, 3), None, None, None, None, 1, True, True, False),
    ],
)
def test_lof(
    forward_func,
    embeddings,
    n_neighbors,
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
        explainer = LofLime(
            forward_func=forward_func,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
        )

        attr = explainer.attribute(
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
        else:
            assert tuple(attr.shape) == (8, 15)

    with pytest.raises(Exception) if fails else nullcontext():
        explainer = LofKernelShap(
            forward_func=forward_func,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
        )

        attr = explainer.attribute(
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
        else:
            assert tuple(attr.shape) == (8, 15)
