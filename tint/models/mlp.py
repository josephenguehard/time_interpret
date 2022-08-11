import torch as th
import torch.nn as nn

from typing import Union


NORMS = {
    "batch_norm_1d": nn.BatchNorm1d,
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


class MLP(nn.Module):
    """
    Base MLP class.

    For more insights into specific arguments of the CNN, please refer
    to `pytorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear>`_.


    Args:
        units (list): A list of units, which creates the layers.
            Default to ``None``
        bias (list, bool): Whether to add bias to each layer.
            Default to ``True``
        dropout (list, float): Dropout rates. Default to 0.0
        norm (list, str): Normalisation layers. Either a list or a string.
            Default to ``None``
        activations (list, str): Activation functions. Either a list or a
            string. Default to ``'relu'``
        activation_final (str): Final activation. Default to ``None``

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    Examples:
        >>> import torch.nn as nn
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> mlp = MLP(units=[5, 10, 1])  # Simple fc with relu activations.
        >>> mlp = MLP(units=[5, 10, 1], dropout=.1)  # Adding dropout.
        >>> mlp = MLP(units=[5, 10, 1], activations="elu")  # Elu activations.
    """

    def __init__(
        self,
        units: list,
        bias: Union[list, bool] = True,
        dropout: Union[list, float] = 0.0,
        norm: Union[list, str] = None,
        activations: Union[list, str] = "relu",
        activation_final: str = None,
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

        layers = [nn.Linear] * length
        if isinstance(bias, bool):
            bias = [bias] * length
        if isinstance(dropout, float):
            dropout = [dropout] * (length - 1)
        if isinstance(norm, str):
            norm = [NORMS[norm]] * (length - 1)
        if isinstance(activations, str):
            activations = [ACTIVATIONS[activations]] * (length - 1)
        if isinstance(activation_final, str):
            activation_final = ACTIVATIONS[activation_final]

        model = dict()
        for i in range(length):
            final_layer = i == length - 1
            name = layers[i].__name__
            model[f"{name}_{i}"] = layers[i](units[i], units[i + 1], bias=bias)

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

            if final_layer and activation_final is not None:
                name = activation_final.__class__.__name__
                model[f"{name}_{i}"] = activation_final

        self.mlp = nn.Sequential()
        for k, v in model.items():
            self.mlp.add_module(k, v)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mlp(x)
