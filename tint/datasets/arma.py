import numpy as np
import os
import pickle as pkl
import torch as th

from .dataset import DataModule

try:
    from statsmodels.tsa.arima_process import ArmaProcess
except ImportError:
    ArmaProcess = None


file_dir = os.path.dirname(__file__)


class Arma(DataModule):
    """
    Arma dataset.

    Args:
        idx (int): Which experiment to run.
        times (int): Length of each time series. Default to 50
        features (int): Number of features in each time series. Default to 50
        ar (list): Coefficient for auto-regressive lag polynomial, including
            zero lag. If ``None``, use default values. Default to ``None``
        ma (list): Coefficient for moving-average lag polynomial, including
            zero lag. If ``None``, use default values. Default to ``None``
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        prop_val (float): Proportion of validation. Default to .2
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42

    References:
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.ArmaProcess.html
        https://arxiv.org/abs/2106.05303
    """

    def __init__(
        self,
        idx: int,
        times: int = 50,
        features: int = 50,
        subset: int = 5,
        ar: list = None,
        ma: list = None,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "arma",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            num_workers=num_workers,
            seed=seed,
        )

        self.idx = idx
        self.times = times
        self.features = features
        self.subset = subset
        self.ar = ar or np.array([2, 0.5, 0.2, 0.1])
        self.ma = ma or np.array([2])

    def download(
        self,
        train_size: int = 1000,
        test_size: int = 100,
        split: str = "train",
    ):
        assert (
            ArmaProcess is not None
        ), "You must install statsmodels to generate arma data."
        file = os.path.join(self.data_dir, f"{split}.npz")

        if split == "train":
            exp = train_size
        elif split == "test":
            exp = test_size
        else:
            raise NotImplementedError

        # Generate data
        data_arma = ArmaProcess(ar=self.ar, ma=self.ma).generate_sample(
            nsample=(exp, self.times, self.features),
            axis=1,
        )

        with open(file, "wb") as fp:
            pkl.dump(obj=data_arma, file=fp)

    def preprocess(self, split: str = "train") -> (th.Tensor, th.Tensor):
        file = os.path.join(self.data_dir, f"{split}.npz")

        # Load data
        with open(file, "rb") as fp:
            features = pkl.load(file=fp)

        # There is no labels here, we just return a tenor of zeros
        return th.Tensor(features[self.idx]), th.zeros(len(features[self.idx]))

    def true_saliency(self, split: str = "train", dim: int = 1) -> th.Tensor:
        file = os.path.join(self.data_dir, f"{split}.npz")

        # Load data
        with open(file, "rb") as fp:
            features = pkl.load(file=fp)

        outputs = th.zeros_like(features)

        if dim == 1:
            # Create a fixed permutation for each experiment
            perm = th.randperm(self.features)

            outputs[
                int(self.times / 4) : int(3 * self.times / 4),
                perm[: self.subset],
            ] = 1

        elif dim == 2:
            t_rand = th.randint(
                low=0, high=self.times - self.subset, size=(1,)
            )
            outputs[
                t_rand : t_rand + self.subset,
                int(self.subset / 4) : int(3 * self.subset / 4),
            ] = 1

        else:
            raise NotImplementedError("dim must be 1 or 2")

        return outputs[self.idx]

    def white_box(
        self, split: str = "train", dim: int = 1
    ) -> (th.Tensor, th.Tensor):
        """
        Create a white box regressor to be interpreted.

        Args:
            split (str): Which split to use: train or test.
                Default to ``'train'``
            dim: On which feature to create a subset of the data.
                Default to 1

        Returns:
            th.Tensor: Output data.
        """
        file = os.path.join(self.data_dir, f"{split}.npz")

        # Load data
        with open(file, "rb") as fp:
            features = pkl.load(file=fp)

        true_saliency = self.true_saliency(split=split, dim=dim)

        outputs = th.zeros(features.shape)

        # Populate the features
        outputs[true_saliency.bool()] = features[true_saliency.bool()]

        outputs = (outputs**2).sum(dim=-1)
        return outputs, true_saliency
