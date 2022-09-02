import torch as th
import torch.nn as nn
import torch.nn.functional as F


TIME_DIM = 1


class TransformerEncoder(nn.Module):
    """
    A base transformer encoder model class.

    Args:
        d_model (int): Input size of the model.
        nhead (int): Number of heads. Default to 1
        dim_feedforward (int): Dimension of the feedforward network model.
            Default to 32
        num_layers (int): Number of layers. Default to 1
        dropout (float): Dropout rates. Default to 0.0
        activation (str): Activation function. Default to ``'relu'``
        layer_norm_eps (float): Eps value in layer normalization components.
            Default to 1e-5
        norm_first (bool): If ``True``, layer norm is done prior to attention
            and feedforward operations, respectively. Default to ``False``
        enable_nested_tensor (bool): If ``True``, input will automatically
            convert to nested tensor. Default to ``False``
        many_to_one (bool): Whether to reduce the temporal dimension.
            Default to ``False``

    References:
        https://pytorch.org/docs/stable/nn.html#transformer-layers

    Examples:
        >>> from tint.models import TransformerEncoder
        <BLANKLINE>
        >>> transformer = TransformerEncoder(10)
        >>> transformer = TransformerEncoder(10, nhead=2, dropout=0.1)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 1,
        dim_feedforward: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        enable_nested_tensor: bool = False,
        many_to_one: bool = False,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor,
        )

        self.many_to_one = many_to_one
        self._size = 1

    @property
    def src_mask(self):
        """
        Generate a square mask for the sequence. The masked positions are
        filled with float('-inf'). Unmasked positions are filled with
        float(0.0).

        Returns:
            th.Tensor: A mask.
        """
        mask = (th.triu(th.ones(self._size, self._size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Update size given inputs
        self._size = x.shape[1]

        # Apply self attention
        out = self.transformer_encoder(
            src=x,
            mask=self.src_mask,
        )

        # Normalize outputs
        out = F.normalize(out, dim=-1, p=2)

        # If many_to_one, reduce temporal dimension
        if self.many_to_one:
            out = out.sum(TIME_DIM)

        return out
