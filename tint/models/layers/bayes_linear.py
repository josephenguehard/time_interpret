import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BayesLinear(nn.Linear):
    r"""
    Applies Bayesian Linear.

    Args:
        prior_sigma (Float): Sigma of prior normal distribution.

    References:
        https://github.com/Harry24k/bayesian-neural-network-pytorch
        https://arxiv.org/abs/2107.02425
    """

    def __init__(
        self,
        prior_sigma: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.prior_log_sigma = math.log(prior_sigma)
        self._freeze = False

        self.weight_log_sigma = nn.Parameter(
            th.Tensor(out_features, in_features)
        )
        if bias:
            self.bias_log_sigma = nn.Parameter(th.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if hasattr(self, "bias"):
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False

    def forward(self, x):
        if self._freeze:
            return F.linear(x, self.weight, self.bias)

        weight = self.weight + th.exp(self.weight_log_sigma) * th.randn_like(
            self.weight_log_sigma
        )

        if hasattr(self, "bias"):
            bias = self.bias + th.exp(self.bias_log_sigma) * th.randn_like(
                self.bias_log_sigma
            )
        else:
            bias = None

        return F.linear(x, weight, bias)

    def loss(self, reduction: str = "mean") -> th.Tensor:
        """
        Compute KL loss for the BayesLinear layer.

        Args:
            reduction: Which type of reduction to apply. Either ``'mean'`` or
                ``'sum'``. Default to ``'mean'``

        Returns:
            th.Tensor: The loss.
        """
        kl = _kl_loss(
            self.weight, self.weight_log_sigma, 0., self.prior_log_sigma,
        )
        n = len(self.weight.view(-1))

        if hasattr(self, "bias"):
            kl += _kl_loss(
                self.bias, self.bias_log_sigma, 0., self.prior_log_sigma,
            )
            n += len(self.bias.view(-1))

        if reduction == "mean":
            return kl / n
        elif reduction == "sum":
            return kl
        else:
            raise ValueError(reduction + " is not valid")


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1):
    """
    An method for calculating KL divergence between two Normal distribution.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    kl = log_sigma_1 - log_sigma_0 + \
         (th.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (
                     2 * math.exp(log_sigma_1) ** 2) - 0.5
    return kl.sum()
