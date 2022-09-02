import pytest
import torch as th

from contextlib import nullcontext

from tint.metrics.white_box import roc_auc


@pytest.mark.parametrize(
    [
        "attributions",
        "true_attributions",
        "normalize",
        "fails",
    ],
    [
        (th.rand(8, 5, 3), th.randint(2, (8, 5, 3)), False, False),
        (th.rand(8, 5, 3), th.randint(2, (8, 5, 3)), True, False),
    ],
)
def test_roc_auc(
    attributions,
    true_attributions,
    normalize,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        roc_auc_ = roc_auc(
            attributions=attributions,
            true_attributions=true_attributions,
            normalize=normalize,
        )
        assert isinstance(roc_auc_, float)
