import os
import pytorch_lightning as pl
import torch as th

from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
    random_split,
    Subset,
)


class Dataset(TorchDataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))


class DataModule(pl.LightningDataModule):
    """
    Base class for datasets.

    Args:
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        prop_val (float): Proportion of validation. Default to .2
        n_folds (int): Number of folds for cross validation. If ``None``,
            the dataset is only split once between train and val using
            ``prop_val``. Default to ``None``
        fold (int): Index of the fold to use with cross-validation.
            Ignored if n_folds is None. Default to ``None``
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prop_val = prop_val
        self.n_folds = n_folds
        self.fold = fold
        self.num_workers = num_workers
        self.seed = seed

        if n_folds is not None:
            assert 0 <= fold < n_folds, "fold must be between 0 and n_folds"

        self.train = None
        self.val = None
        self.test = None
        self.predict = None

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def collate_fn(batch: list) -> (th.Tensor, th.Tensor):
        return (
            th.stack([b["x"] for b in batch]),
            th.stack([b["y"] for b in batch]),
        )

    def download(self, split: str = "train"):
        raise NotImplementedError

    def preprocess(self, split: str = "train") -> dict:
        raise NotImplementedError

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "train.npz")):
            self.download("train")
        if not os.path.exists(os.path.join(self.data_dir, "test.npz")):
            self.download("test")

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            full = Dataset(self.preprocess("train"))

            if self.n_folds is None:
                len_val = int(len(full) * self.prop_val)
                self.train, self.val = random_split(
                    full,
                    [len(full) - len_val, len_val],
                    th.Generator().manual_seed(self.seed),
                )

            else:
                kf = KFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.seed
                )
                all_splits = [k for k in kf.split(full)]
                train_indexes, val_indexes = all_splits[self.fold]
                train_indexes, val_indexes = (
                    train_indexes.tolist(),
                    val_indexes.tolist(),
                )

                self.train = Subset(full, train_indexes)
                self.val = Subset(full, val_indexes)

        if stage == "test" or stage is None:
            self.test = Dataset(self.preprocess("test"))

        if stage == "predict" or stage is None:
            self.predict = Dataset(self.preprocess("test"))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_data(self, split: str = "train"):
        assert split in ["train", "val", "test"], "Unknown split."
        return getattr(getattr(self, split), "data")
