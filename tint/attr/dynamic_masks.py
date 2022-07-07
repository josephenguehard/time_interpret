import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _format_output,
    _is_tuple,
    _reduce_list,
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
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            mask_net (MaskNet): A Mask model. If ``None``, a default model
                will be provided. Default to ``None``

        Returns:
            th.Tensor: Attributions.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_input(inputs)

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)

        # Get representations
        outputs_list = list()
        for x in inputs:
            outputs_list.append(
                self._attributes(
                    inputs=x,
                    trainer=trainer,
                    mask_net=mask_net,
                ).reshape(-1, 1)
            )
        outputs = _reduce_list(outputs_list)

        return _format_output(is_inputs_tuple, outputs)

    @staticmethod
    def _attributes(
        inputs: th.Tensor,
        trainer: Trainer,
        mask_net: MaskNet = None,
        batch_size: int = 32,
    ):
        # Get input and output shape
        shape = inputs.shape

        # Init MaskNet if not provided
        if mask_net is None:
            mask_net = MaskNet()
        else:
            mask_net = copy.deepcopy(mask_net)

        # Init model
        mask_net.net[0].init(
            shape=shape,
            n_epochs=trainer.max_epochs,
        )

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(inputs, inputs), batch_size=batch_size
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        return mask_net.net[0].representation()
