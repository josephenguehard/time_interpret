import pytest

from tint.datasets import Hawkes
from tint.datasets.dataset import DataModule


@pytest.mark.parametrize(
    [
        "mu",
        "alpha",
        "decay",
        "window",
        "batch_size",
        "prop_val",
        "n_folds",
        "fold",
        "num_workers",
        "seed",
    ],
    [
        (None, None, None, None, 32, 0.1, 5, 0, 0, 42),
        ([0.05, 0.1], None, None, None, 32, 0.1, 5, 0, 0, 42),
        (None, None, None, None, 64, 0.1, 5, 0, 0, 42),
        (None, None, None, None, 32, 0.3, 5, 0, 0, 42),
        (None, None, None, None, 32, 0.1, 8, 2, 0, 42),
        (None, None, None, None, 32, 0.1, 5, 0, 3, 12),
    ],
)
def test_init(
    mu,
    alpha,
    decay,
    window,
    batch_size,
    prop_val,
    n_folds,
    fold,
    num_workers,
    seed,
):
    hawkes = Hawkes(
        mu=mu,
        alpha=alpha,
        decay=decay,
        window=window,
        batch_size=batch_size,
        prop_val=prop_val,
        n_folds=n_folds,
        fold=fold,
        num_workers=num_workers,
        seed=seed,
    )
    assert isinstance(hawkes, DataModule)
