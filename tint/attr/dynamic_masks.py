import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

from .models import MaskNet


class DynaMask(PerturbationAttribution):
    """
    Dynamic masks method.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://arxiv.org/pdf/2106.05303
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        trainer: Trainer = None,
        mask_net: MaskNet = None,
        batch_size: int = 32,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            mask_net (MaskNet): A Mask model. If ``None``, a default model
                will be provided. Default to ``None``
            batch_size (int): Batch size for Mask training. Default to 32

        Returns:
            (th.Tensor, tuple): Attributions.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_input(inputs)

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)
        else:
            trainer = copy.deepcopy(trainer)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"

        # Get input and output shape
        shape = (min(batch_size, len(inputs[0])),) + inputs[0].shape[1:]

        # Init MaskNet if not provided
        if mask_net is None:
            mask_net = MaskNet(forward_func=self.forward_func)
        else:
            mask_net = copy.deepcopy(mask_net)

        # Init model
        mask_net.net.init(
            shape=shape,
            n_epochs=trainer.max_epochs,
        )

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(inputs[0], inputs[0]), batch_size=batch_size
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        # Set model to eval mode
        mask_net.eval()

        # Get attributions as mask representation
        attributions = (mask_net.net.representation(),)

        # Format attributions and return
        return _format_output(is_inputs_tuple, attributions)
