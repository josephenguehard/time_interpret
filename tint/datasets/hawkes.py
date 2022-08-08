import pickle as pkl
import numpy as np
import os
import torch as th

from torch.nn.utils.rnn import pad_sequence

from .dataset import DataModule


try:
    from tick.hawkes import SimuHawkes, HawkesKernelExp
except ImportError:
    SimuHawkes = None
    HawkesKernelExp = None


file_dir = os.path.dirname(__file__)


class Hawkes(DataModule):
    """
    Hawkes dataset.

    Args:
        mu (list): Intensity baselines. If ``None``, use default values.
            Default to ``None``
        alpha (list): Events parameters. If ``None``, use default values.
            Default to ``None``
        decay (list): Intensity decays. If ``None``, use default values.
            Default to ``None``
        window (int): The window of the simulated process. If ``None``, use
            default value. Default to ``None``
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
        mu: list = None,
        alpha: list = None,
        decay: list = None,
        window: int = None,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "hawkes",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )

        self.mu = mu or [0.05, 0.05]
        self.alpha = alpha or [[0.1, 0.2], [0.2, 0.1]]
        self.decay = decay or [[1.0, 1.0], [1.0, 1.0]]
        self.window = window or 1000

    def download(
        self,
        train_size: int = 1000,
        test_size: int = 100,
        split: str = "train",
    ):
        assert (
            SimuHawkes is not None
        ), "You must install tick to generate hawkes data."
        file = os.path.join(self.data_dir, f"{split}.npz")

        if split == "train":
            idx = range(train_size)
        elif split == "test":
            idx = range(train_size, train_size + test_size)
        else:
            raise NotImplementedError

        points = [
            self.generate_points(
                mu=self.mu,
                alpha=self.alpha,
                decay=self.decay,
                window=self.window,
                seed=i,
            )
            for i in idx
        ]

        with open(file, "wb") as fp:
            pkl.dump(obj=points, file=fp)

    def preprocess(self, split: str = "train") -> dict:
        file = os.path.join(self.data_dir, f"{split}.npz")

        # Load data
        with open(file, "rb") as fp:
            data = pkl.load(file=fp)

        # Create features
        features = pad_sequence(
            [self.get_features(x) for x in data],
            batch_first=True,
        ).unsqueeze(-1)

        # Create labels
        labels = pad_sequence(
            [self.get_labels(x) for x in data],
            batch_first=True,
        ).unsqueeze(-1)

        return {"x": features, "y": labels}

    @staticmethod
    def generate_points(
        mu: list,
        alpha: list,
        decay: list,
        window: int,
        seed: int,
        dt: float = 0.01,
    ):
        """
        Generates points of an marked Hawkes processes using the tick library.

        Args:
            mu (list): Hawkes baseline.
            alpha (list): Event parameter.
            decay (list): Decay parameter.
            window (int): The window of the simulated process.
            seed (int): The random seed.
            dt (float): Granularity. Default to 0.01
        """
        hawkes = SimuHawkes(
            n_nodes=len(mu), end_time=window, verbose=False, seed=seed
        )
        for i in range(len(mu)):
            for j in range(len(mu)):
                hawkes.set_kernel(
                    i=i,
                    j=j,
                    kernel=HawkesKernelExp(
                        intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]
                    ),
                )
            hawkes.set_baseline(i, mu[i])

        hawkes.track_intensity(dt)
        hawkes.simulate()
        return hawkes.timestamps

    def true_saliency(
        self,
        t: th.Tensor,
        mu: th.Tensor = None,
        alpha: th.Tensor = None,
        decay: th.Tensor = None,
        times: th.Tensor = None,
        labels: th.Tensor = None,
        split: str = "train",
    ):
        """
        Compute the true saliency given some time queries.

        B: Batch size.
        T: Temporal dim.
        N: Number of processes.
        Q: Number of time queries.

        Args:
            t (th.Tensor): Time queries. Shape Q
            mu (th.Tensor): Intensity baselines. Shape N, Values 0..1
            alpha (th.Tensor): Events parameters. Shape N x N, Values 0..1
            decay (th.Tensor): Intensity decays. Shape N x N, Values 0..1
            times (th.Tensor): Times of the process. Shape B x T x 1
            labels (th.Tensor: Labels of the process. Shape B x T x 1
            split (str): Data split. Default to ``'train'``

        Returns:
            th.Tensor: true_saliency
        """
        # Get params
        mu = th.Tensor(self.mu) if mu is None else mu
        alpha = th.Tensor(self.alpha) if alpha is None else alpha
        decay = th.Tensor(self.decay) if decay is None else decay

        # Get data
        if times is None or labels is None:
            data = self.preprocess(split=split)
            times, labels = data["x"], data["y"]

        # Compute influence of each element
        t = t.unsqueeze(0).unsqueeze(0)

        diff = (times - t) * (times > 0) * (times < t)
        exp = (th.exp(diff) * (times > 0) * (times < t)).float()

        labelled_exp = th.zeros(exp.shape + (len(mu),)).scatter(
            -1,
            labels.unsqueeze(-1).repeat(1, 1, t.shape[-1], 1),
            exp.unsqueeze(-1),
        )

        true_saliency = th.matmul(labelled_exp, decay)
        true_saliency = th.matmul(true_saliency, alpha)
        true_saliency = true_saliency.sum(-1) / true_saliency.sum(-1).sum(
            1, keepdim=True
        )

        # We set eventual nans to zeros due to the division
        return true_saliency.nan_to_num_()

    @staticmethod
    def intensity(
        mu: th.Tensor,
        alpha: th.Tensor,
        decay: th.Tensor,
        times: th.Tensor,
        labels: th.Tensor,
        t: th.Tensor,
    ) -> th.Tensor:
        """
        Given parameters mu, alpha and decay, some
        times and labels, and a vector of query times t,
        compute intensities at these time points.

        B: Batch size.
        T: Temporal dim.
        N: Number of processes.
        Q: Number of time queries.

        Args:
            mu (th.Tensor): Intensity baselines. Shape N, Values 0..1
            alpha (th.Tensor): Events parameters. Shape N x N, Values 0..1
            decay (th.Tensor): Intensity decays. Shape N x N, Values 0..1
            times (th.Tensor): Times of the process. Shape B x T x 1
            labels (th.Tensor: Labels of the process. Shape B x T x 1
            t (th.Tensor): Query times. Shape Q

        Returns:
            th.Tensor: Intensities. Shape B x Q x N
        """
        t = t.unsqueeze(0).unsqueeze(0)

        diff = (times - t) * (times > 0) * (times < t)
        exp = (th.exp(diff) * (times > 0) * (times < t)).float()

        labelled_exp = th.zeros(exp.shape + (len(mu),)).scatter(
            -1,
            labels.unsqueeze(-1).repeat(1, 1, t.shape[-1], 1),
            exp.unsqueeze(-1),
        )

        sum_ = th.matmul(labelled_exp, decay).sum(1)

        return th.matmul(sum_, alpha) + mu

    @staticmethod
    def get_features(point: list) -> th.Tensor:
        """
        Create features and labels from a hawkes process.

        Args:
            point (list): A hawkes process.
        """
        times = np.concatenate(point)
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        return th.from_numpy(times)

    @staticmethod
    def get_labels(point: list) -> th.Tensor:
        """
        Create features and labels from a hawkes process.

        Args:
            point (list): A hawkes process.
        """
        times = np.concatenate(point)
        labels = np.concatenate([[i] * len(x) for i, x in enumerate(point)])
        sort_idx = np.argsort(times)
        labels = labels[sort_idx]
        return th.from_numpy(labels)
