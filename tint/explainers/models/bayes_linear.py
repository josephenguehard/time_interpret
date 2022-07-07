import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from captum._utils.models import LinearModel
from captum._utils.models.linear_model.train import (
    sgd_train_linear_model,
    sklearn_train_linear_model,
    l2_loss,
)
from torch.utils.data import DataLoader
from typing import Callable


class BayesLinearModel(LinearModel):
    r"""
    Bayesian Linear model.

    Built on top of the captum linear layer to be integrated in this library.

    Args:
        train_fn (callable): The function to train with.

    References:
        https://github.com/Harry24k/bayesian-neural-network-pytorch
        https://arxiv.org/abs/2107.02425
    """

    def __init__(self, train_fn: Callable, **kwargs) -> None:
        super().__init__(train_fn=train_fn, **kwargs)

        self.prior_log_sigma = None
        self.weight_log_sigma = None
        self.bias_log_sigma = None
        self._freeze = False
        self.has_bias = True

    def _construct_model_params(
        self,
        prior_sigma: float = 0.1,
        in_features: int = None,
        out_features: int = None,
        norm_type: str = None,
        affine_norm: bool = False,
        bias: bool = True,
        weight_values: th.Tensor = None,
        bias_value: th.Tensor = None,
        classes: th.Tensor = None,
    ):
        r"""
        Lazily initializes a linear model. This will be called for you in a
        train method.

        Args:
            in_features (int):
                The number of input features
            out_features (int):
                The number of output features.
            norm_type (str):
                The type of normalization that can occur. Please assign this
                to one of `PyTorchLinearModel.SUPPORTED_NORMS`.
            affine_norm (bool):
                Whether or not to learn an affine transformation of the
                normalization parameters used.
            bias (bool):
                Whether to add a bias term. Not needed if normalized input.
            weight_values (th.Tensor):
                The values to initialize the linear model with. This must be a
                1D or 2D tensor, and of the form `(num_outputs, num_features)`
                or `(num_features,)`. Additionally, if this is provided you
                need not to provide `in_features` or `out_features`.
            bias_value (th.Tensor): The bias value to initialize the model
                with. Default to ``None``
            classes (th.Tensor):
                The list of prediction classes supported by the model in case
                it performs classification. In case of regression it is set to
                None. Default to ``None``
        """
        super()._construct_model_params(
            in_features=in_features,
            out_features=out_features,
            norm_type=norm_type,
            affine_norm=affine_norm,
            bias=bias,
            weight_values=weight_values,
            bias_value=bias_value,
            classes=classes,
        )

        self.prior_log_sigma = math.log(prior_sigma)
        self._freeze = False

        self.weight_log_sigma = nn.Parameter(
            th.Tensor(out_features, in_features)
        )

        if bias:
            self.bias_log_sigma = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter("bias_log_sigma", None)

        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        self.bias_log_sigma.data.fill_(self.prior_log_sigma)

        self.has_bias = bias

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert self.linear is not None

        if self.norm is not None:
            x = self.norm(x)

        if self._freeze:
            return F.linear(x, self.linear.weight, self.linear.bias)

        weight = self.linear.weight + th.exp(
            self.weight_log_sigma
        ) * th.randn_like(self.weight_log_sigma)

        if self.has_bias:
            bias = self.linear.bias + th.exp(
                self.bias_log_sigma
            ) * th.randn_like(self.bias_log_sigma)
        else:
            bias = None

        return F.linear(x, weight, bias)

    def kl_loss(self, reduction: str = "mean") -> th.Tensor:
        """
        Compute KL loss for the BayesLinear layer.

        Args:
            reduction: Which type of reduction to apply. Either ``'mean'`` or
                ``'sum'``. Default to ``'mean'``

        Returns:
            th.Tensor: The loss.
        """
        kl = _kl_loss(
            self.linear.weight,
            self.weight_log_sigma,
            0.0,
            self.prior_log_sigma,
        )
        n = len(self.weight.view(-1))

        if hasattr(self, "bias"):
            kl += _kl_loss(
                self.linear.bias,
                self.bias_log_sigma,
                0.0,
                self.prior_log_sigma,
            )
            n += len(self.linear.bias.view(-1))

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
    kl = (
        log_sigma_1
        - log_sigma_0
        + (th.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2)
        / (2 * math.exp(log_sigma_1) ** 2)
        - 0.5
    )
    return kl.sum()


class SGDBayesLinearModel(BayesLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""
        Factory class. Construct a `BayesLinearModel` with the
        `sgd_train_linear_model` as the train method

        Args:
            kwargs
                Arguments send to `self._construct_model_params` after
                `self.fit` is called. Please refer to that method for parameter
                documentation.
        """
        super().__init__(train_fn=sgd_train_linear_model, **kwargs)


class SGDBayesLasso(SGDBayesLinearModel):
    def __init__(self, kl_weight: float = 0.1, **kwargs) -> None:
        r"""
        Factory class to train a `LinearModel` with SGD
        (`sgd_train_linear_model`) whilst setting appropriate parameters to
        optimize for ridge regression loss. This optimizes L2 loss + alpha * L1
        regularization.

        Please note that with SGD it is not guaranteed that weights will
        converge to 0.
        """
        super().__init__(**kwargs)

        self.kl_weight = kl_weight

    def fit(self, train_data: DataLoader, **kwargs):
        return super().fit(
            train_data=train_data,
            loss_fn=lambda *args: l2_loss(*args)
            + self.kl_weight * self.kl_loss(),
            reg_term=1,
            **kwargs,
        )


class SGDBayesRidge(SGDBayesLinearModel):
    def __init__(self, kl_weight: float = 0.1, **kwargs) -> None:
        r"""
        Factory class to train a `LinearModel` with SGD
        (`sgd_train_linear_model`) whilst setting appropriate parameters to
        optimize for ridge regression loss. This optimizes L2 loss + alpha *
        L2 regularization.
        """
        super().__init__(**kwargs)

        self.kl_weight = kl_weight

    def fit(self, train_data: DataLoader, **kwargs):
        return super().fit(
            train_data=train_data,
            loss_fn=lambda *args: l2_loss(*args)
            + self.kl_weight * self.kl_loss(),
            reg_term=2,
            **kwargs,
        )


class SGDBayesLinearRegression(SGDBayesLinearModel):
    def __init__(self, kl_weight: float = 0.1, **kwargs) -> None:
        r"""
        Factory class to train a `LinearModel` with SGD
        (`sgd_train_linear_model`). For linear regression this assigns the loss
        to L2 and no regularization.
        """
        super().__init__(**kwargs)

        self.kl_weight = kl_weight

    def fit(self, train_data: DataLoader, **kwargs):
        return super().fit(
            train_data=train_data,
            loss_fn=lambda *args: l2_loss(*args)
            + self.kl_weight * self.kl_loss(),
            reg_term=None,
            **kwargs,
        )


class SkLearnBayesLinearModel(BayesLinearModel):
    def __init__(self, sklearn_module: str, **kwargs) -> None:
        r"""
        Factory class to construct a `LinearModel` with sklearn training method.

        Please note that this assumes:

        0. You have sklearn and numpy installed
        1. The dataset can fit into memory

        SkLearn support does introduce some slight overhead as we convert the
        tensors to numpy and then convert the resulting trained model to a
        `LinearModel` object. However, this conversion should be negligible.

        Args:
            sklearn_module
                The module under sklearn to construct and use for training, e.g.
                use "svm.LinearSVC" for an SVM or "linear_model.Lasso" for Lasso.

                There are factory classes defined for you for common use cases,
                such as `SkLearnLasso`.
            kwargs
                The kwargs to pass to the construction of the sklearn model
        """
        super().__init__(train_fn=sklearn_train_linear_model, **kwargs)

        self.sklearn_module = sklearn_module

    def fit(self, train_data: DataLoader, **kwargs):
        r"""
        Args:
            train_data
                Train data to use
            kwargs
                Arguments to feed to `.fit` method for sklearn
        """
        return super().fit(
            train_data=train_data,
            sklearn_trainer=self.sklearn_module,
            **kwargs,
        )


class SkLearnARDRegression(BayesLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""
        Factory class. Trains a model with `sklearn.linear_model.ARDRegression`.

        Any arguments provided to the sklearn constructor can be provided
        as kwargs here.
        """
        super().__init__(sklearn_module="linear_model.ARDRegression", **kwargs)

    def fit(self, train_data: DataLoader, **kwargs):
        return super().fit(train_data=train_data, **kwargs)


class SkLearnBayesianRidge(BayesLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""
        Factory class. Trains a model with `sklearn.linear_model.BayesianRidge`.

        Any arguments provided to the sklearn constructor can be provided
        as kwargs here.
        """
        super().__init__(sklearn_module="linear_model.BayesianRidge", **kwargs)

    def fit(self, train_data: DataLoader, **kwargs):
        return super().fit(train_data=train_data, **kwargs)
