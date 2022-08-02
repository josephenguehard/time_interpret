import torch as th
import torch.nn as nn

from torch.distributions import MultivariateNormal, kl_divergence
from typing import Callable, Union

from tint.models import Net


class BayesMask(nn.Module):
    def __init__(self, forward_func: Callable) -> None:
        super().__init__()
        self.forward_func = forward_func

        self.input_size = None
        self.register_parameter("mean", None)
        self.register_parameter("cov", None)

    def init(self, input_size: tuple) -> None:
        self.input_size = input_size

        self.mean = nn.Parameter(th.Tensor(*input_size))

        cov = th.Tensor(*input_size + (input_size[-1],))
        self.cov = nn.Parameter(th.matmul(cov, cov.transpose(-2, -1)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mean.fill_(0.5)
        self.cov.data = th.eye(self.input_size[-1]).expand(
            self.input_size + (self.input_size[-1],)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        normal = MultivariateNormal(loc=self.mean, covariance_matrix=self.cov)
        sample = normal.rsample()

        x[sample < 0.5] = 0.0

        return self.forward_func(x)

    def regularisation(self, loss: th.Tensor) -> th.Tensor:
        # Get uninformative mean and cov
        mean = th.zeros(*self.input_size)
        cov = th.eye(self.input_size[-1])
        cov = cov.expand(self.input_size + (self.input_size[-1],))

        # Compute kl divergence between normals
        normal = MultivariateNormal(loc=self.mean, covariance_matrix=self.cov)
        normal_ = MultivariateNormal(loc=mean, covariance_matrix=cov)
        kl = kl_divergence(normal, normal_)

        # Return loss + regularisation
        return loss + kl.sum()

    def clamp(self):
        self.mean.data.clamp(0, 1)
        self.cov.data.clamp(0)


class BayesMaskNet(Net):
    def __init__(
        self,
        forward_func: Callable,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = BayesMask(forward_func=forward_func)

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def step(self, batch):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, self.net.forward_func(y))
        return loss

    def training_step_end(self, step_output):
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        # Clamp mask
        self.net.clamp()

        return step_output
