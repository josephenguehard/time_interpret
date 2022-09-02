import pytest
import torch as th

from contextlib import nullcontext
from pytorch_lightning import Trainer

from tint.attr import Retain
from tint.attr.models import RetainNet


@pytest.mark.parametrize(
    ["retain", "datamodule", "labels", "batch_size", "fails"],
    [
        (None, None, th.randint(2, (8, 10)), 32, False),
        (RetainNet(), None, th.rand(8, 10, 2), 32, False),
        (
            RetainNet(),
            None,
            th.rand(
                8,
            ),
            32,
            True,
        ),
        (RetainNet(temporal_labels=False), None, th.rand(8, 2), 32, False),
    ],
)
def test_init(retain, datamodule, labels, batch_size, fails):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Retain(
            retain=retain,
            datamodule=datamodule,
            features=th.rand(8, 10, 3),
            labels=labels,
            trainer=Trainer(max_steps=1),
            batch_size=batch_size,
        )
        assert isinstance(explainer, Retain)


@pytest.mark.parametrize(
    [
        "retain",
        "datamodule",
        "labels",
        "batch_size",
        "inputs",
        "target",
        "return_temporal_attributions",
        "fails",
    ],
    [
        (None, None, th.randint(2, (8, 10)), 32, th.rand(8, 10, 3), None, False, False),
        (None, None, th.randint(2, (8, 10)), 32, th.rand(8, 10, 3), th.randint(2, (8, 10)), False, False),
        (None, None, th.randint(2, (8, 10)), 32, th.rand(8, 10, 3), None, True, False),
    ],
)
def test_retain(
    retain,
    datamodule,
    labels,
    batch_size,
    inputs,
    target,
    return_temporal_attributions,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Retain(
            retain=retain,
            datamodule=datamodule,
            features=th.rand(8, 10, 3),
            labels=labels,
            trainer=Trainer(max_steps=1),
            batch_size=batch_size,
        )

        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            return_temporal_attributions=return_temporal_attributions,
        )
        if return_temporal_attributions:
            assert tuple(attr.shape) == (inputs.shape[0], inputs.shape[1]) + inputs.shape[1:]
        else:
            assert attr.shape == inputs.shape
