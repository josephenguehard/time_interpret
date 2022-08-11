import torch as th
import torch.nn as nn

from typing import Union


NORMS = {
    "batch_norm_2d": nn.BatchNorm2d,
}


ACTIVATIONS = {
    "celu": nn.CELU(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(),
    "log_softmax": nn.LogSoftmax(dim=-1),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "tanh": nn.Tanh(),
    "tanhshrink": nn.Tanhshrink(),
}


POOLS = {
    "max_pool_2d": nn.MaxPool2d(kernel_size=2),
}


class CNN(nn.Module):
    r"""
    Base CNN class.

    For more insights into specific arguments of the CNN, please refer
    to `pytorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.

    Args:
        units (list): A list of units, which creates the layers.
            Default to ``None``
        dropout (list, float): Dropout rates. Default to 0.0
        norm (list, str): Normalisation layers. Either a list or a string.
            Default to ``None``
        activations (list, str): Activation functions. Either a list or a
            string. Default to ``'relu'``
        pooling (list, str): Pooling module. Either a list or a string.
            Default t0 ``None``

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    Examples:
        >>> import torch.nn as nn
        >>> from tint.models import CNN
        <BLANKLINE>
        >>> cnn = CNN(units=[10, 8, 6], kernel_size=3)  # Simple cnn with relu activations.
        >>> cnn = CNN(units=[10, 8, 6], kernel_size=3, dropout=.1)  # Adding dropout.
        >>> cnn = CNN(units=[10, 8, 6], kernel_size=3, activations="elu")  # Elu activations.
    """

    def __init__(
        self,
        units: list,
        kernel_size: Union[list, int],
        stride: Union[list, int] = 1,
        padding: Union[list, int] = 0,
        dilation: Union[list, int] = 1,
        groups: Union[list, int] = 1,
        bias: Union[list, bool] = True,
        padding_mode: Union[list, str] = "zeros",
        dropout: Union[list, float] = 0.0,
        norm: Union[list, str] = None,
        activations: Union[list, str] = "relu",
        pooling: Union[list, str] = None,
    ):
        super().__init__()

        assert len(units) > 1, "At least two units must be provided."
        length = len(units) - 1

        if isinstance(dropout, list):
            assert len(dropout) == length - 1, (
                f"Inconsistent number of dropout: found "
                f"{len(dropout)} but should be {length - 1}."
            )
        if isinstance(norm, list):
            assert len(norm) == length - 1, (
                f"Inconsistent number of norm: found "
                f"{len(norm)} but should be {length - 1}."
            )
        if isinstance(activations, list):
            assert len(activations) == length - 1, (
                f"Inconsistent number of activations: found "
                f"{len(activations)} but should be {length - 1}."
            )
        if isinstance(pooling, list):
            assert len(pooling) == length - 1, (
                f"Inconsistent number of pooling: found "
                f"{len(pooling)} but should be {length - 1}."
            )

        layers = [nn.Conv2d] * length
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * length
        if isinstance(stride, int):
            stride = [stride] * length
        if isinstance(padding, int):
            padding = [padding] * length
        if isinstance(dilation, int):
            dilation = [dilation] * length
        if isinstance(groups, int):
            groups = [groups] * length
        if isinstance(bias, bool):
            bias = [bias] * length
        if isinstance(padding_mode, str):
            padding_mode = [padding_mode] * length
        if isinstance(dropout, float):
            dropout = [dropout] * (length - 1)
        if isinstance(norm, str):
            norm = [NORMS[norm]] * (length - 1)
        if isinstance(activations, str):
            activations = [ACTIVATIONS[activations]] * (length - 1)
        if isinstance(pooling, str):
            pooling = [POOLS[pooling]] * (length - 1)

        model = dict()
        for i in range(length):
            final_layer = i == length - 1
            name = layers[i].__name__
            model[f"{name}_{i}"] = layers[i](
                units[i],
                units[i + 1],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=dilation[i],
                groups=groups[i],
                bias=bias[i],
                padding_mode=padding_mode[i],
            )

            if not final_layer and dropout[i] > 0:
                model[f"Dropout_{i}"] = nn.Dropout(p=dropout[i])

            if not final_layer and norm is not None and norm[i] is not None:
                name = norm[i].__name__
                model[f"{name}_{i}"] = norm[i](num_features=units[i + 1])

            if (
                not final_layer
                and activations is not None
                and activations[i] is not None
            ):
                name = activations[i].__class__.__name__
                model[f"{name}_{i}"] = activations[i]

            if (
                not final_layer
                and pooling is not None
                and pooling[i] is not None
            ):
                name = pooling[i].__class__.__name__
                model[f"{name}_{i}"] = pooling[i]

            if final_layer:
                name = nn.Flatten.__name__
                model[f"{name}_{i}"] = nn.Flatten(1)

        self.cnn = nn.Sequential()
        for k, v in model.items():
            self.cnn.add_module(k, v)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn(x)
