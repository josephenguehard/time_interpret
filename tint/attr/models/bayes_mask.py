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
    """
    Bayes Mask model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        distribution (str): Which distribution to use in order to draw the
            mask. Either ``'none'``, ``'bernoulli'``, ``'normal'`` or
            ``'gumbel_softmax'``. Default to ``'bernoulli'``
        hard (bool): Whether to create hard mask values or not. In both
            cases, soft values will be used for back-propagation.
            Default to ``True``
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        eps (float): Optional param for normal distribution.
            Set the target covariance matrix to eps * th.eye().
            Default to 1e-3
        batch_size (int): Batch size of the model. Default to 32
    """

    def __init__(
        self,
        forward_func: Callable,
        distribution: str = "bernoulli",
        hard: bool = True,
        model: nn.Module = None,
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
        object.__setattr__(self, "forward_func", forward_func)
        self.distribution = distribution
        self.hard = hard
        self.model = model
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
        cov = th.zeros(self.input_size + (shape,)).to(tril.device)
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
            y_hard = (
                th.zeros_like(samples)
                .to(samples.device)
                .scatter_(-1, index, 1.0)
            )
            samples = y_hard - samples.detach() + samples

        # Mask data according to samples
        # We eventually cut samples up to x time dimension
        x *= samples[:, : x.shape[1], ...]

        # We forward model if provided
        if self.model is not None:
            x = self.model(x)

        # Return f(perturbed x)
        return _run_forward(
            forward_func=self.forward_func,
            inputs=x,
            additional_forward_args=additional_forward_args,
        )

    def regularisation(self, loss: th.Tensor) -> th.Tensor:
        # Get uninformative mean and tril
        mean = th.zeros_like(self.mean).to(self.mean.device)
        if self.distribution == "normal":
            shape = self.input_size[-1]
            tril = th.zeros_like(self.tril).to(self.tril.device)
            tril[
                ...,
                th.tril_indices(shape, shape)[0]
                == th.tril_indices(shape, shape)[1],
            ] = self.eps

        # Return loss + regularisation
        loss += (self.mean - mean).abs().mean()
        if self.distribution == "normal":
            loss += (self.tril - tril).abs().mean()

        return loss

    def clamp(self):
        self.mean.data.clamp_(0, 1)
        if self.distribution == "normal":
            self.tril.data.clamp_(self.eps, 1)

    def representation(self):
        return self.mean.detach().cpu().clamp(0, 1)

    def covariance(self):
        assert self.distribution == "normal"
        return self.get_cov(self.tril.detach().cpu())


class BayesMaskNet(Net):
    """
    Bayes mask model as a Pytorch Lightning model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        distribution (str): Which distribution to use in order to draw the
            mask. Either ``'none'``, ``'bernoulli'``, ``'normal'`` or
            ``'gumbel_softmax'``. Default to ``'bernoulli'``
        hard (bool): Whether to create hard mask values or not. In both
            cases, soft values will be used for back-propagation.
            Default to ``True``
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        eps (float): Optional param for normal distribution.
            Set the target covariance matrix to eps * th.eye().
            Default to 1e-3
        batch_size (int): Batch size of the model. Default to 32
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0
    """

    def __init__(
        self,
        forward_func: Callable,
        distribution: str = "bernoulli",
        hard: bool = True,
        model: nn.Module = None,
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
            hard=hard,
            model=model,
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
        loss = self.loss(y_hat, y_target)
        return loss

    def training_step_end(self, step_output):
        # Add regularisation from Mask network
        step_output = self.net.regularisation(step_output)

        return step_output

    def configure_optimizers(self):
        params = [{"params": self.net.mean}]

        if self.net.distribution == "normal":
            params += [{"params": self.net.tril}]

        if self.net.model is not None:
            params += [{"params": self.net.model.parameters()}]

        if self._optim == "adam":
            optim = th.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}
