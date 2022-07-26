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
from captum._utils.typing import TensorOrTupleOfTensorsGeneric, TargetType

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

from .models import RetainNet


class Retain(PerturbationAttribution):
    """
    Retain explainer method.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://arxiv.org/pdf/1608.05745
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        trainer: Trainer = None,
        retain_net: RetainNet = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            target (int, tuple, tensor, list): Output indices. Default to
                ``None``
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            retain_net (RetainNet): A Retain model. If ``None``, a default model
                will be provided. Default to ``None``

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

        # Get representations
        outputs_list = list()
        for x in inputs:
            outputs_list.append(
                self._attributes(
                    inputs=x,
                    target=target,
                    trainer=trainer,
                    retain_net=retain_net,
                ).reshape(-1, 1)
            )
        outputs = _reduce_list(outputs_list)

        return _format_output(is_inputs_tuple, outputs)

    @staticmethod
    def _attributes(
        inputs: th.Tensor,
        target: th.Tensor,
        trainer: Trainer,
        retain_net: RetainNet = None,
        batch_size: int = 32,
    ):
        # Init MaskNet if not provided
        if retain_net is None:
            retain_net = RetainNet()
        else:
            retain_net = copy.deepcopy(retain_net)

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(inputs, inputs), batch_size=batch_size
        )

        # Fit model
        trainer.fit(retain_net, train_dataloaders=dataloader)

        return retain_net.net[0].representation(inputs=inputs, target=target)
