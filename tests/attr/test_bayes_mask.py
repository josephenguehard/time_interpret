import pytest
import torch as th

from contextlib import nullcontext
from pytorch_lightning import Trainer

from tint.attr import BayesMask
from tint.attr.models import BayesMaskNet
from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    ["forward_func", "fails"],
    [
        (BasicModel(), False),
        (BasicModel5_MultiArgs(), False),
    ],
)
def test_init(forward_func, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesMask(
            forward_func=forward_func,
        )
        assert isinstance(explainer, BayesMask)


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "additional_forward_args",
        "mask_net",
        "batch_size",
        "temporal_additional_forward_args",
        "return_temporal_attributions",
        "return_covariance",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, 8, None, False, False, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            8,
            (True, True),
            False,
            False,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), None, None, 4, None, False, False, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, 8, None, True, False, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, 8, None, False, True, False),
    ],
)
def test_bayes_mask(
    forward_func,
    inputs,
    additional_forward_args,
    mask_net,
    batch_size,
    temporal_additional_forward_args,
    return_temporal_attributions,
    return_covariance,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = BayesMask(forward_func=forward_func)

        # Just one step for rapid testing
        trainer = Trainer(max_steps=1)

        # If return covariance, change distribution to normal
        if return_covariance:
            mask_net = BayesMaskNet(forward_func, distribution="normal")

        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            trainer=trainer,
            mask_net=mask_net,
            batch_size=batch_size,
            temporal_additional_forward_args=temporal_additional_forward_args,
            return_temporal_attributions=return_temporal_attributions,
            return_covariance=return_covariance,
        )
        if return_covariance:
            attr = attr[0]
        if return_temporal_attributions:
            assert tuple(attr.shape) == (inputs.shape[0], inputs.shape[1]) + inputs.shape[1:]
        else:
            assert attr.shape == inputs.shape
