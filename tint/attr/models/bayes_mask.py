import torch as th
import torch.nn as nn
import torch.nn.functional as F

from captum._utils.common import _run_forward

from torch.distributions import (
    ContinuousBernoulli,
    MultivariateNormal,
    kl_divergence,
)
from typing import Callable, Union

from tint.models import Net


class BayesMask(nn.Module):
    def __init__(
        self,
        forward_func: Callable,
        distribution: str = "bernoulli",
        eps: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        assert distribution in ["bernoulli", "normal"], (
            f"distribution should be either bernoulli or normal, "
            f"got {distribution}."
        )

        super().__init__()
        self.forward_func = forward_func
        self.distribution = distribution
        self.eps = eps
        self.batch_size = batch_size

        self.input_size = None
        self.register_parameter("mean", None)
        if self.distribution == "normal":
            self.register_parameter("tril", None)

    def init(self, input_size: tuple, batch_size: int = 32) -> None:
        self.input_size = input_size
        self.batch_size = batch_size

        self.mean = nn.Parameter(th.Tensor(*input_size))

        if self.distribution == "normal":
            self.tril = nn.Parameter(
                th.Tensor(
                    *input_size[:-1]
                    + (input_size[-1] * (input_size[-1] + 1) // 2,)
                )
            )

        self.reset_parameters()

    def get_cov(self, tril):
        shape = self.input_size[-1]
        cov = th.zeros(self.input_size + (shape,))
        cov[
            ...,
            th.tril_indices(shape, shape)[0],
            th.tril_indices(shape, shape)[1],
        ] = tril
        return cov

    def reset_parameters(self) -> None:
        self.mean.data.fill_(0.5)

        if self.distribution == "normal":
            shape = self.input_size[-1]
            self.tril.data = th.zeros_like(self.tril.data)
            self.tril.data[
                ...,
                th.tril_indices(shape, shape)[0]
                == th.tril_indices(shape, shape)[1],
            ] = 1.0

    def forward(
        self, x: th.Tensor, batch_idx, *additional_forward_args
    ) -> th.Tensor:
        # Clamp mask
        self.clamp()

        # Sample from distribution
        if self.distribution == "bernoulli":
            dist = ContinuousBernoulli(probs=self.mean)
        elif self.distribution == "normal":
            dist = MultivariateNormal(
                loc=self.mean, scale_tril=self.get_cov(self.tril)
            )
        else:
            raise NotImplementedError
        samples = dist.rsample()

        # Subset sample to current batch
        samples = samples[
            self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
        ]

        # The threshold we use is 0.5, so we set 1 - s as
        # the opposite logit
        samples = th.stack([samples, 1. - samples], dim=-1)

        # We use the Gumbel-Max trick to discretize the samples
        samples = F.gumbel_softmax(samples, tau=0.01, hard=True, dim=-1)[..., 0]

        # Mask data according to samples
        x *= samples

        # Return f(perturbed x)
        return _run_forward(
            forward_func=self.forward_func,
            inputs=x,
            additional_forward_args=additional_forward_args,
        )

    def regularisation(self, loss: th.Tensor) -> th.Tensor:
        # Get uninformative mean and tril
        mean = th.zeros_like(self.mean)
        if self.distribution == "normal":
            shape = self.input_size[-1]
            tril = th.zeros_like(self.tril)
            tril[
                ...,
                th.tril_indices(shape, shape)[0]
                == th.tril_indices(shape, shape)[1],
            ] = self.eps

        # Compute kl divergence between distributions
        if self.distribution == "bernoulli":
            dist = ContinuousBernoulli(probs=self.mean)
            target = ContinuousBernoulli(probs=mean)
        elif self.distribution == "normal":
            dist = MultivariateNormal(
                loc=self.mean, scale_tril=self.get_cov(self.tril)
            )
            target = MultivariateNormal(
                loc=mean, scale_tril=self.get_cov(tril)
            )
        else:
            raise NotImplementedError
        kl = kl_divergence(target, dist)

        # Return loss + regularisation
        return loss + kl.sum()

    def clamp(self):
        self.mean.data.clamp_(0, 1)
        if self.distribution == "normal":
            self.tril.data.clamp_(self.eps, 1)

    def representation(self):
        return self.mean.detach().cpu()

    def covariance(self):
        assert self.distribution == "normal"
        return self.get_cov(self.tril.detach().cpu())


class BayesMaskNet(Net):
    def __init__(
        self,
        forward_func: Callable,
        distribution: str = "bernoulli",
        eps: float = 1e-3,
        batch_size: int = 32,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = BayesMask(
            forward_func=forward_func,
            distribution=distribution,
            eps=eps,
            batch_size=batch_size,
        )

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get perturbed output
        if additional_forward_args is None:
            y_hat = self(x.float(), batch_idx)
        else:
            y_hat = self(x.float(), batch_idx, *additional_forward_args)

        # Get unperturbed output
        y_target = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )

        # Compute loss
        loss = self._loss(y_hat, y_target)
        return loss

    def training_step_end(self, step_output):
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        return step_output
