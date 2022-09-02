import pytest

from tint.datasets import HMM
from tint.datasets.dataset import DataModule


@pytest.mark.parametrize(
    [
        "n_signal",
        "n_state",
        "corr_features",
        "imp_features",
        "scale",
        "p0",
        "batch_size",
        "prop_val",
        "n_folds",
        "fold",
        "num_workers",
        "seed",
    ],
    [
        (3, 1, None, None, None, None, 32, 0.1, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 64, 0.1, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 32, 0.3, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 32, 0.1, 8, 2, 0, 42),
        (3, 1, None, None, None, None, 32, 0.1, 5, 0, 3, 12),
    ],
)
def test_init(
    n_signal,
    n_state,
    corr_features,
    imp_features,
    scale,
    p0,
    batch_size,
    prop_val,
    n_folds,
    fold,
    num_workers,
    seed,
):
    hmm = HMM(
        n_signal=n_signal,
        n_state=n_state,
        corr_features=corr_features,
        imp_features=imp_features,
        scale=scale,
        p0=p0,
        batch_size=batch_size,
        prop_val=prop_val,
        n_folds=n_folds,
        fold=fold,
        num_workers=num_workers,
        seed=seed,
    )
    assert isinstance(hmm, DataModule)


@pytest.mark.parametrize(
    [
        "n_signal",
        "n_state",
        "corr_features",
        "imp_features",
        "scale",
        "p0",
        "batch_size",
        "prop_val",
        "n_folds",
        "fold",
        "num_workers",
        "seed",
    ],
    [
        (3, 1, None, None, None, None, 32, 0.1, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 64, 0.1, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 32, 0.3, 5, 0, 0, 42),
        (3, 1, None, None, None, None, 32, 0.1, 8, 2, 0, 42),
        (3, 1, None, None, None, None, 32, 0.1, 5, 0, 3, 12),
    ],
)
def test_hmm(
    n_signal,
    n_state,
    corr_features,
    imp_features,
    scale,
    p0,
    batch_size,
    prop_val,
    n_folds,
    fold,
    num_workers,
    seed,
):
    hmm = HMM(
        n_signal=n_signal,
        n_state=n_state,
        corr_features=corr_features,
        imp_features=imp_features,
        scale=scale,
        p0=p0,
        batch_size=batch_size,
        prop_val=prop_val,
        n_folds=n_folds,
        fold=fold,
        num_workers=num_workers,
        seed=seed,
    )
    hmm.download(split="test", test_size=10, signal_length=20)
    x_test = hmm.preprocess(split="test")["x"]
    y_test = hmm.preprocess(split="test")["y"]
    assert tuple(x_test.shape) == (10, 20, 3)
    assert tuple(y_test.shape) == (10, 20)

    true_saliency = hmm.true_saliency("test")
    assert tuple(true_saliency.shape) == (10, 20, 3)
