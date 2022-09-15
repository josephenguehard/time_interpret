import pytest
import torch as th

from contextlib import nullcontext

from tint.metrics.white_box import mse


@pytest.mark.parametrize(
    [
        "attributions",
        "true_attributions",
        "normalize",
        "fails",
    ],
    [
        (th.rand(8, 5, 3), th.rand(8, 5, 3), False, False),
        (th.rand(8, 5, 3), th.rand(8, 5, 3), True, False),
    ],
)
def test_mse(
    attributions,
    true_attributions,
    normalize,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        mse_ = mse(
            attributions=attributions,
            true_attributions=true_attributions,
            normalize=normalize,
        )
        assert isinstance(mse_, float)
