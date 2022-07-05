import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


RNNS = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


class RNN(nn.Module):
    """
    A base recurrent model class.

    Args:
        input_size (int): Input size of the model.
        rnn (nn.RNNBase, str): Which type of RNN to use. Default to ``'rnn'``
        hidden_size (int): The number of features in the hidden state h.
            Default to 32
        num_layers (int): Number of recurrent layers. Default to 1
        bias (bool): Whether to use bias. Default to ``True``
        dropout (float): Dropout rates. Default to 0.0
        bidirectional (bool): If ``True``, becomes a bidirectional RNN.
            Default to ``False``
    """

    def __init__(
        self,
        input_size: int,
        rnn: Union[nn.RNNBase, str] = "rnn",
        hidden_size: int = 32,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.rnn = RNNS[rnn](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Flatten parameters due to saving issue with pickle
        self.rnn.flatten_parameters()

        # Forward, normalize and add results to inputs
        out, _ = self.rnn(x)
        return F.normalize(out, dim=-1, p=2)
