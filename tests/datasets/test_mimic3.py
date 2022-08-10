import pytest

from tint.datasets import Mimic3
from tint.datasets.dataset import DataModule


@pytest.mark.parametrize(
    [
        "batch_size",
        "prop_val",
        "n_folds",
        "fold",
        "num_workers",
        "seed",
    ],
    [
        (32, 0.1, 5, 0, 0, 42),
        (32, 0.1, 5, 0, 0, 42),
        (64, 0.1, 5, 0, 0, 42),
        (32, 0.3, 5, 0, 0, 42),
        (32, 0.1, 8, 2, 0, 42),
        (32, 0.1, 5, 0, 3, 12),
    ],
)
def test_init(batch_size, prop_val, n_folds, fold, num_workers, seed):
    mimic3 = Mimic3(
        batch_size=batch_size,
        prop_val=prop_val,
        n_folds=n_folds,
        fold=fold,
        num_workers=num_workers,
        seed=seed,
    )
    assert isinstance(mimic3, DataModule)
