import os
import pytorch_lightning as pl
import torch as th

from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from typing import Callable


class Dataset(TorchDataset):
    def __init__(
        self,
        data,
        target,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    """
    Base class for datasets.

    Args:
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        prop_val (float): Proportion of validation. Default to .2
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        prop_val: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prop_val = prop_val
        self.num_workers = num_workers
        self.seed = seed

        self.train = None
        self.val = None
        self.test = None
        self.predict = None

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def preprocess(self, split: str = "train") -> (th.Tensor, th.Tensor):
        raise NotImplementedError

    def download(self, split: str = "train"):
        raise NotImplementedError

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "train.npz")):
            self.download("train")
        if not os.path.exists(os.path.join(self.data_dir, "test.npz")):
            self.download("test")

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            full = Dataset(*self.preprocess("train"))

            len_val = int(len(full) * self.prop_val)
            self.train, self.val = random_split(
                full,
                [len(full) - len_val, len_val],
                th.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.test = Dataset(*self.preprocess("test"))

        if stage == "predict" or stage is None:
            self.predict = Dataset(*self.preprocess("test"))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_data(self, split: str = "train"):
        assert split in ["train", "val", "test"], "Unknown split."
        return getattr(getattr(self, split), "data")
