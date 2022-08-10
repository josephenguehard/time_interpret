import torch as th
import torch.nn as nn
import torch.nn.functional as F

from captum._utils.common import _run_forward

from torch.distributions import (
    ContinuousBernoulli,
    MultivariateNormal,
)
from typing import Callable, Union

from tint.models import Net

EPS = 1e-7


class BayesMask(nn.Module):
    def __init__(
        self,
        forward_func: Callable,
        distribution: str = "bernoulli",
        hard: bool = True,
        eps: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        assert distribution in [
            "none",
            "bernoulli",
            "normal",
            "gumbel_softmax",
        ], (
            f"distribution should be either none, bernoulli, "
            f"normal or gumbel_softmax, got {distribution}."
        )

        super().__init__()
        self.forward_func = forward_func
        self.distribution = distribution
        self.hard = hard
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
        if self.distribution == "none":
            samples = self.mean

        elif self.distribution == "bernoulli":
            dist = ContinuousBernoulli(probs=self.mean)
            samples = dist.rsample()

        elif self.distribution == "normal":
            dist = MultivariateNormal(
                loc=self.mean, scale_tril=self.get_cov(self.tril)
            )
            samples = dist.rsample()

        elif self.distribution == "gumbel_softmax":
            samples = self.mean

            # The threshold we use is 0.5, so we set 1 - s as
            # the opposite logit
            samples = th.stack([samples, 1.0 - samples], dim=-1)

            # We compute the inverse of the softmax so the logits passed to
            # the gumbel softmax are accurate
            samples = th.log(samples + EPS) + th.logsumexp(
                samples, -1, keepdim=True
            )

            samples = F.gumbel_softmax(samples, tau=0.01, dim=-1)[..., 0]

        else:
            raise NotImplementedError

        # Subset sample to current batch
        samples = samples[
            self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
        ]

        if self.hard:
            # We subtract and add samples
            # By detaching one, the gradients are equivalent to just passing
            # soft samples.
            index = samples.max(-1, keepdim=True)[1]
            y_hard = th.zeros_like(samples).scatter_(-1, index, 1.0)
            samples = y_hard - samples.detach() + samples

        # Mask data according to samples
        # We eventually cut samples up to x time dimension
        x *= samples[:, : x.shape[1], ...]

        # Return f(perturbed x)
        with th.autograd.set_grad_enabled(False):
            y_hat = _run_forward(
                forward_func=self.forward_func,
                inputs=x,
                additional_forward_args=additional_forward_args,
            )
        return y_hat

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

        # Return loss + regularisation
        loss += (self.mean - mean).mean()
        if self.distribution == "normal":
            loss += (self.tril - tril).mean().abs()

        return loss

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
        hard: bool = True,
        eps: float = 1e-3,
        batch_size: int = 32,
        temporal: bool = False,
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
            hard=hard,
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
        self.temporal = temporal

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, *additional_forward_args = batch

        if self.temporal:
            t = th.randint(x.shape[1], (1,)).item()
            x = x[:, : t + 1, ...]
            y = y[:, : t + 1, ...]

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
        with th.autograd.set_grad_enabled(False):
            y_target = _run_forward(
                forward_func=self.net.forward_func,
                inputs=y,
                additional_forward_args=tuple(additional_forward_args)
                if additional_forward_args is not None
                else None,
            )

        # If loss is cross_entropy, take softmax of y_target
        if isinstance(self._loss, nn.CrossEntropyLoss):
            y_target = y_target.softmax(-1)

        # Compute loss
        loss = self._loss(y_hat, y_target)
        return loss

    def training_step_end(self, step_output):
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        return step_output
