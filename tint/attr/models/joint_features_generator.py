import torch as th
import torch.nn as nn

from typing import Union

from tint.models import Net


class JointFeatureGenerator(nn.Module):
    """
    Conditional generator model to predict future observations.

    Args:
        rnn_hidden_size (int): Size of hidden units for the recurrent
            structure. Default to 100
        dist_hidden_size (int): Size of the distribution hidden units.
            Default to 10
        latent_size: Size of the latent distribution. Default to 100

    References:
        https://proceedings.neurips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html
    """

    def __init__(
        self,
        rnn_hidden_size: int = 100,
        dist_hidden_size: int = 10,
        latent_size: int = 100,
    ):
        super(JointFeatureGenerator, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.dist_hidden_size = dist_hidden_size
        self.latent_size = latent_size

        self.register_module("rnn", None)
        self.register_module("dist_predictor", None)
        self.register_module("cov_generator", None)
        self.register_module("mean_generator", None)

        self.feature_size = None

    def init(self, feature_size):
        # Generates the parameters of the distribution
        self.rnn = nn.GRU(feature_size, self.rnn_hidden_size, batch_first=True)
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.normal_(self.rnn.__getattr__(p), 0.0, 0.02)

        self.dist_predictor = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, self.latent_size * 2),
            nn.Tanh(),
        )

        self.cov_generator = nn.Sequential(
            nn.Linear(self.latent_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, feature_size**2),
            nn.ReLU(),
        )

        self.mean_generator = nn.Sequential(
            nn.Linear(self.latent_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, feature_size),
        )

        self.feature_size = feature_size

    def likelihood_distribution(self, past: th.Tensor):
        all_encoding, encoding = self.rnn(past)
        h = encoding.view(encoding.size(1), -1)

        # Find the distribution of the latent variable Z
        mu_std = self.dist_predictor(h)
        mu = mu_std[:, : mu_std.shape[1] // 2]
        std = mu_std[:, mu_std.shape[1] // 2 :]

        # sample Z from the distribution
        z = mu + std * th.randn_like(mu)

        # Generate the distribution P(X|H,Z)
        mean = self.mean_generator(z)
        cov_noise = (
            th.eye(self.feature_size).unsqueeze(0).repeat(len(z), 1, 1) * 1e-5
        )
        a = self.cov_generator(z).view(
            -1, self.feature_size, self.feature_size
        )
        covariance = th.bmm(a, th.transpose(a, 1, 2)) + cov_noise
        return mean, covariance

    def forward(self, past: th.Tensor):
        mean, covariance = self.likelihood_distribution(past)
        likelihood = th.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=covariance,
        )
        return likelihood.rsample()

    def forward_conditional(
        self,
        past: th.Tensor,
        current: th.Tensor,
        sig_inds: list,
    ):
        if current.shape[-1] == len(sig_inds):
            return current, current
        if len(current.shape) == 1:
            current = current.unsqueeze(0)

        # Compute mean and covariance of X_t given the past
        mean, covariance = self.likelihood_distribution(past)  # P(X_t|X_0:t-1)

        # Get explored and ignored features indices
        sig_inds_comp = list(set(range(past.shape[-1])) - set(sig_inds))
        ind_len = len(sig_inds)
        ind_len_not = len(sig_inds_comp)

        x_ind = current[:, sig_inds].view(-1, ind_len)
        mean_1 = mean[:, sig_inds_comp].view(-1, ind_len_not)
        cov_1_2 = covariance[:, sig_inds_comp, :][:, :, sig_inds].view(
            -1, ind_len_not, ind_len
        )
        cov_2_2 = covariance[:, sig_inds, :][:, :, sig_inds].view(
            -1, ind_len, ind_len
        )
        cov_1_1 = covariance[:, sig_inds_comp, :][:, :, sig_inds_comp].view(
            -1, ind_len_not, ind_len_not
        )

        mean_cond = mean_1 + th.bmm(
            (th.bmm(cov_1_2, th.inverse(cov_2_2))),
            (x_ind - mean[:, sig_inds]).view(-1, ind_len, 1),
        ).squeeze(-1)
        covariance_cond = cov_1_1 - th.bmm(
            th.bmm(cov_1_2, th.inverse(cov_2_2)), th.transpose(cov_1_2, 2, 1)
        )

        # P(x_{-i,t}|x_{i,t})
        likelihood = th.distributions.multivariate_normal.MultivariateNormal(
            loc=mean_cond.squeeze(-1), covariance_matrix=covariance_cond
        )
        sample = likelihood.rsample()

        full_sample = current.clone()
        full_sample[:, sig_inds_comp] = sample

        return full_sample, mean[:, sig_inds_comp]


class JointFeatureGeneratorNet(Net):
    """
    Conditional generator model to predict future observations as a
    Pytorch Lightning module.

    Args:
        rnn_hidden_size (int): Size of hidden units for the recurrent
            structure. Default to 100
        dist_hidden_size (int): Size of the distribution hidden units.
            Default to 10
        latent_size: Size of the latent distribution. Default to 100
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    References:
        https://proceedings.neurips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html
    """

    def __init__(
        self,
        rnn_hidden_size: int = 100,
        dist_hidden_size: int = 10,
        latent_size: int = 100,
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        generator = JointFeatureGenerator(
            rnn_hidden_size=rnn_hidden_size,
            dist_hidden_size=dist_hidden_size,
            latent_size=latent_size,
        )

        super().__init__(
            layers=generator,
            loss=None,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def step(self, batch, stage, t):  # noqa
        x = batch[0]
        mean, covariance = self.net.likelihood_distribution(x[:, :t, ...])
        dist = th.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=covariance,
        )
        loss = -dist.log_prob(x[:, t, ...]).mean()
        return loss

    def training_step(self, batch, batch_idx):
        t = th.randint(low=4, high=batch[0].shape[1], size=(1,)).item()
        loss = self.step(batch=batch, stage="train", t=t)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        t = th.randint(low=4, high=batch[0].shape[1], size=(1,)).item()
        loss = self.step(batch=batch, stage="val", t=t)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        t = batch[0].shape[1] - 1
        loss = self.step(batch=batch, stage="test", t=t)
        self.log("test_loss", loss)
