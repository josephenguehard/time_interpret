import pytest
import torch as th

from contextlib import nullcontext

from tint.metrics.white_box import aur


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
def test_aur(
    attributions,
    true_attributions,
    normalize,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        aur_ = aur(
            attributions=attributions,
            true_attributions=true_attributions,
            normalize=normalize,
        )
        assert isinstance(aur_, float)
