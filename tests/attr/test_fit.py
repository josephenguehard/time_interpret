import pytest
import torch as th

from contextlib import nullcontext
from pytorch_lightning import Trainer

from tint.attr import Fit
from tint.attr.models import JointFeatureGeneratorNet
from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "generator", "datamodule", "batch_size", "fails"],
    [
        (BasicModel(), None, None, 32, False),
        (BasicModel5_MultiArgs(), None, None, 32, False),
        (BasicModel(), JointFeatureGeneratorNet(), None, 32, False),
    ],
)
def test_init(forward_func, generator, datamodule, batch_size, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Fit(
            forward_func=forward_func,
            generator=generator,
            datamodule=datamodule,
            features=th.rand(8, 5, 3),
            trainer=Trainer(max_steps=1),
            batch_size=batch_size,
        )
        assert isinstance(explainer, Fit)


@pytest.mark.parametrize(
    [
        "forward_func",
        "generator",
        "datamodule",
        "batch_size",
        "inputs",
        "additional_forward_args",
        "n_samples",
        "distance_metric",
        "multilabel",
        "temporal_additional_forward_args",
        "return_temporal_attributions",
        "show_progress",
        "fails",
    ],
    [
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "kl", False, None, False, False, False),
        (
            BasicModel5_MultiArgs(),
            None,
            None,
            32,
            th.rand(8, 5, 3),
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            10,
            "kl",
            False,
            (True, True),
            False,
            False,
            False,
        ),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 5, "kl", False, None, False, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "mean_divergence", False, None, False, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "LHS", False, None, False, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "RHS", False, None, False, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "wrong", False, None, False, False, True),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "kl", True, None, False, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "kl", False, None, True, False, False),
        (BasicModel(), None, None, 32, th.rand(8, 5, 3), None, 10, "kl", False, None, False, True, False),
    ],
)
def test_fit(
    forward_func,
    generator,
    datamodule,
    batch_size,
    inputs,
    additional_forward_args,
    n_samples,
    distance_metric,
    multilabel,
    temporal_additional_forward_args,
    return_temporal_attributions,
    show_progress,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Fit(
            forward_func=forward_func,
            generator=generator,
            datamodule=datamodule,
            features=th.rand(8, 5, 3),
            trainer=Trainer(max_steps=1),
            batch_size=batch_size,
        )

        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            distance_metric=distance_metric,
            multilabel=multilabel,
            temporal_additional_forward_args=temporal_additional_forward_args,
            return_temporal_attributions=return_temporal_attributions,
            show_progress=show_progress,
        )
        if return_temporal_attributions:
            assert tuple(attr.shape) == (inputs.shape[0], inputs.shape[1]) + inputs.shape[1:]
        else:
            assert attr.shape == inputs.shape
