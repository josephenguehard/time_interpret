import torch as th
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Callable, Union

from tint.models import Net


def masked_softmax(batch_tensor, mask):
    exp = th.exp(batch_tensor)
    masked_exp = exp * mask
    sum_masked_exp = th.sum(masked_exp, dim=1, keepdim=True)
    return masked_exp / sum_masked_exp


class Retain(nn.Module):
    """
    RETAIN network.

    Args:
        dim_emb (int): Dimension of the embedding. Default to 128
        dropout_input (float): Dropout rate for the input. Default to 0.8
        dropout_emb (float): Dropout of the embedding. Default to 0.5
        dim_alpha (int): Hidden size of the alpha rnn. Default to 128
        dim_beta (int): Hidden size of the beta rnn. Default to 128
        dropout_context (float): Dropout rate of the context vector.
            Default to 0.5
        dim_output (int): Size of the output. Default to 2
        temporal_labels (bool): Whether to use temporal labels or
            static labels. Default to ``True``

    References:
        `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_
    """

    def __init__(
        self,
        dim_emb: int = 128,
        dropout_input: float = 0.8,
        dropout_emb: float = 0.5,
        dim_alpha: int = 128,
        dim_beta: int = 128,
        dropout_context: float = 0.5,
        dim_output: int = 2,
        temporal_labels: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input),
            nn.LazyLinear(dim_emb, bias=False),
            nn.Dropout(p=dropout_emb),
        )

        self.rnn_alpha = nn.GRU(
            input_size=dim_emb,
            hidden_size=dim_alpha,
            num_layers=1,
            batch_first=True,
        )

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        nn.init.xavier_normal_(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(
            input_size=dim_emb,
            hidden_size=dim_beta,
            num_layers=1,
            batch_first=True,
        )

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        nn.init.xavier_normal_(
            self.beta_fc.weight,
            gain=nn.init.calculate_gain("tanh"),
        )
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output),
        )
        nn.init.xavier_normal_(self.output[1].weight)
        self.output[1].bias.data.zero_()

        self.temporal_labels = temporal_labels

    def forward(self, x, lengths):
        batch_size, max_len = x.size()[:2]

        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)

        packed_input = pack_padded_sequence(
            emb,
            lengths,
            batch_first=True,
        )
        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=True)

        # mask -> batch_size X max_len X 1
        mask = Variable(
            th.FloatTensor(
                [
                    [1.0 if i < lengths[idx] else 0.0 for i in range(max_len)]
                    for idx in range(batch_size)
                ]
            )
            .unsqueeze(2)
            .to(x.device),
            requires_grad=False,
        )

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=True)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        beta = th.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = th.bmm(th.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit, alpha, beta


class RetainNet(Net):
    """
    Retain Network as a Pytorch Lightning module.

    Args:
        dim_emb (int): Dimension of the embedding. Default to 128
        dropout_input (float): Dropout rate for the input. Default to 0.8
        dropout_emb (float): Dropout of the embedding. Default to 0.5
        dim_alpha (int): Hidden size of the alpha rnn. Default to 128
        dim_beta (int): Hidden size of the beta rnn. Default to 128
        dropout_context (float): Dropout rate of the context vector.
            Default to 0.5
        dim_output (int): Size of the output. Default to 2
        temporal_labels (bool): Whether to use temporal labels or
            static labels. Default to ``True``
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    References:
        `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_

    Examples:
        >>> from tint.attr.models import RetainNet
        <BLANKLINE>
        >>> retain = RetainNet(
        ...     dim_emb=128,
        ...     dropout_emb=0.4,
        ...     dim_alpha=8,
        ...     dim_beta=8,
        ...     dropout_context=0.4,
        ...     dim_output=2,
        ...     loss="cross_entropy",
        ... )
    """

    def __init__(
        self,
        dim_emb: int = 128,
        dropout_input: float = 0.8,
        dropout_emb: float = 0.5,
        dim_alpha: int = 128,
        dim_beta: int = 128,
        dropout_context: float = 0.5,
        dim_output: int = 2,
        temporal_labels: bool = True,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        retain = Retain(
            dim_emb=dim_emb,
            dropout_input=dropout_input,
            dropout_emb=dropout_emb,
            dim_alpha=dim_alpha,
            dim_beta=dim_beta,
            dropout_context=dropout_context,
            dim_output=dim_output,
            temporal_labels=temporal_labels,
        )

        super().__init__(
            layers=retain,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

    def step(self, batch, batch_idx, stage):
        x, y = batch

        lengths = th.randint(low=4, high=x.shape[1], size=(len(x),))
        lengths, _ = th.sort(lengths, descending=True)
        lengths[0] = x.shape[1]

        y_hat, _, _ = self.net(x=x.float(), lengths=lengths)
        if self.net.temporal_labels:
            y = y[th.arange(len(x)), lengths - 1, ...]
        loss = self.loss(y_hat, y)

        return loss
