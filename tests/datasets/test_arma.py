import pytest

from tint.datasets import Arma
from tint.datasets.dataset import DataModule


@pytest.mark.parametrize(
    [
        "times",
        "features",
        "subset",
        "batch_size",
        "prop_val",
        "n_folds",
        "fold",
        "num_workers",
        "seed",
    ],
    [
        (50, 50, 5, 32, 0.1, 5, 0, 0, 42),
        (50, 50, 10, 64, 0.1, 5, 0, 0, 42),
        (50, 50, 5, 32, 0.3, 5, 0, 0, 42),
        (50, 50, 5, 32, 0.1, 8, 2, 0, 42),
        (50, 50, 5, 32, 0.1, 5, 0, 2, 24),
    ],
)
def test_init(
    times,
    features,
    subset,
    batch_size,
    prop_val,
    n_folds,
    fold,
    num_workers,
    seed,
):
    arma = Arma(
        times=times,
        features=features,
        subset=subset,
        batch_size=batch_size,
        prop_val=prop_val,
        n_folds=n_folds,
        fold=fold,
        num_workers=num_workers,
        seed=seed,
    )
    assert isinstance(arma, DataModule)
