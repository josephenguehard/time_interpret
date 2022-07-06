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

        self.weight_log_sigma = nn.Parameter(
            th.Tensor(out_features, in_features)
        )

        if bias:
            self.bias_log_sigma = nn.Parameter(th.Tensor(out_features))

    def reset_parameters(self):
        super().reset_parameters()

        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if hasattr(self, "bias"):
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, x):
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
