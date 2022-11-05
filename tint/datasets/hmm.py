import numpy as np
import os
import pickle as pkl
import torch as th

from .dataset import DataModule


file_dir = os.path.dirname(__file__)


def logit(x):
    # The real logit should be '1.0 / (1 + np.exp(-x))', but we use -2 as in
    # https://github.com/JonathanCrabbe/Dynamask for fair comparison.
    return 1.0 / (1 + np.exp(-2 * x))


class HMM(DataModule):
    """
    2-state Hidden Markov Model as described in the DynaMask paper.

    Args:
        n_signal (int): Number of different signals. Default to 3
        n_state (int): Number of different possible states. Default to 1
        corr_features (list): Features that re correlated with the important
            feature in each state. If ``None``, use default values.
            Default to ``None``
        imp_features (list): Features that are always set as important.
            If ``None``, use default values. Default to ``None``
        scale (list): Scaling factor for distribution mean in each state.
            If ``None``, use default values. Default to ``None``
        p0 (list): Starting probability. If ``None``, use default values.
            Default to ``None``
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

    References:
        https://arxiv.org/pdf/2106.05303

    Examples:
        >>> from tint.datasets import HMM
        <BLANKLINE>
        >>> hmm = HMM()
        >>> hmm.download(split="train")
        >>> x_train = hmm.preprocess(split="train")["x"]
        >>> y_train = hmm.preprocess(split="train")["y"]
    """

    def __init__(
        self,
        n_signal: int = 3,
        n_state: int = 1,
        corr_features: list = None,
        imp_features: list = None,
        scale: list = None,
        p0: list = None,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "hmm",
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

        self.n_signal = n_signal
        self.n_state = n_state

        self.corr_feature = corr_features or [0, 0]
        self.imp_feature = imp_features or [1, 2]
        self.scale = scale or [[0.1, 1.6, 0.5], [-0.1, -0.4, -1.5]]
        self.p0 = p0 or [0.5]

    def init_dist(self):
        # Covariance matrix is constant across states but distribution
        # means change based on the state value
        state_count = 2**self.n_state
        cov = np.eye(self.n_signal) * 0.8

        covariance = []
        for i in range(state_count):
            c = cov.copy()
            c[self.imp_feature[i], self.corr_feature[i]] = 0.01
            c[self.corr_feature[i], self.imp_feature[i]] = 0.01
            c = c + np.eye(self.n_signal) * 1e-3
            covariance.append(c)
        covariance = np.array(covariance)

        mean = []
        for i in range(state_count):
            m = self.scale[i]
            mean.append(m)
        mean = np.array(mean)

        return mean, covariance

    @staticmethod
    def next_state(previous_state, t):
        if previous_state == 1:
            params = 0.95
        else:
            params = 0.05

        params = params - float(t / 500) if params > 0.8 else params
        next_state = np.random.binomial(1, params)
        return next_state

    def download(
        self,
        train_size: int = 800,
        test_size: int = 200,
        signal_length: int = 200,
        split: str = "train",
    ):
        file = os.path.join(self.data_dir, f"{split}_")

        if split == "train":
            count = train_size
        elif split == "test":
            count = test_size
        else:
            raise NotImplementedError

        features = list()
        labels = list()
        importance_score = list()
        all_states = list()
        label_logits = list()
        mean, cov = self.init_dist()

        for i in range(count):
            signal = list()
            states = list()
            y = list()
            importance = list()
            y_logits = list()

            previous = np.random.binomial(1, self.p0)[0]
            delta_state = 0
            state_n = None
            for i in range(signal_length):
                next = self.next_state(previous, delta_state)
                state_n = next

                if state_n == previous:
                    delta_state += 1
                else:
                    delta_state = 0

                # if state_n!=previous:
                imp_sig = np.zeros(3)
                if state_n != previous or i == 0:
                    imp_sig[self.imp_feature[state_n]] = 1

                importance.append(imp_sig)
                sample = np.random.multivariate_normal(
                    mean[state_n],
                    cov[state_n],
                )
                previous = state_n
                signal.append(sample)
                y_logit = logit(sample[self.imp_feature[state_n]])
                y_label = np.random.binomial(1, y_logit)

                y.append(y_label)
                y_logits.append(y_logit)
                states.append(state_n)

            signal = np.array(signal)
            y = np.array(y)
            importance = np.array(importance)

            features.append(signal)
            labels.append(y)
            importance_score.append(importance)
            all_states.append(states)
            label_logits.append(y_logits)

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "wb"
        ) as fp:
            pkl.dump(obj=features, file=fp)
        with open(
            os.path.join(self.data_dir, file + "labels.npz"), "wb"
        ) as fp:
            pkl.dump(obj=labels, file=fp)
        with open(
            os.path.join(self.data_dir, file + "importance.npz"), "wb"
        ) as fp:
            pkl.dump(obj=importance_score, file=fp)
        with open(
            os.path.join(self.data_dir, file + "states.npz"), "wb"
        ) as fp:
            pkl.dump(obj=all_states, file=fp)
        with open(
            os.path.join(self.data_dir, file + "labels_logits.npz"), "wb"
        ) as fp:
            pkl.dump(obj=label_logits, file=fp)

    def preprocess(self, split: str = "train") -> dict:
        file = os.path.join(self.data_dir, f"{split}_")

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "rb"
        ) as fp:
            features = np.stack(pkl.load(file=fp))
        with open(
            os.path.join(self.data_dir, file + "labels.npz"), "rb"
        ) as fp:
            labels = np.stack(pkl.load(file=fp))

        return {
            "x": th.from_numpy(features).float(),
            "y": th.from_numpy(labels).long(),
        }

    def prepare_data(self):
        """"""
        if not os.path.exists(
            os.path.join(self.data_dir, "train_features.npz")
        ):
            self.download(split="train")
        if not os.path.exists(
            os.path.join(self.data_dir, "test_features.npz")
        ):
            self.download(split="test")

    def true_saliency(self, split: str = "train") -> th.Tensor:
        file = os.path.join(self.data_dir, f"{split}_")

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "rb"
        ) as fp:
            features = np.stack(pkl.load(file=fp))

        # Load the true states that define the truly salient features
        # and define A as in Section 3.2:
        with open(
            os.path.join(self.data_dir, file + "states.npz"), "rb"
        ) as fp:
            true_states = np.stack(pkl.load(file=fp))
            true_states += 1

            true_saliency = th.zeros(features.shape)
            for exp_id, time_slice in enumerate(true_states):
                for t_id, feature_id in enumerate(time_slice):
                    true_saliency[exp_id, t_id, feature_id] = 1
            true_saliency = true_saliency.long()

        return true_saliency
