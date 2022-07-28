import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric, TargetType

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from .models import Retain as RetainModel, RetainNet


class Retain(PerturbationAttribution):
    """
    Retain explainer method.

    References:
        https://arxiv.org/pdf/1608.05745
    """

    def __init__(self) -> None:
        super().__init__(forward_func=None)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        trainer: Trainer = None,
        retain_net: RetainNet = None,
        batch_size: int = 32,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            target (int, tuple, tensor, list): Output indices. Default to
                ``None``
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            retain_net (RetainNet): A Retain model. If ``None``, a default
            model will be provided. Default to ``None``
            batch_size (int): Batch size for Retain training. Default to 32

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

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"

        # Make target a tensor
        target = self._format_target(inputs, target)

        # Init MaskNet if not provided
        if retain_net is None:
            retain_net = RetainNet(loss="cross_entropy")
        else:
            retain_net = copy.deepcopy(retain_net)

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(inputs[0], target), batch_size=batch_size
        )

        # Fit model
        trainer.fit(retain_net, train_dataloaders=dataloader)

        # Set model to eval mode
        retain_net.eval()

        # Get attributions
        attributions = (
            self.representation(
                model=retain_net.net,
                inputs=inputs[0],
                target=target,
            ),
        )

        return _format_output(is_inputs_tuple, attributions)

    @staticmethod
    def representation(
        model: RetainModel,
        inputs: th.Tensor,
        target: th.Tensor = None,
    ):
        """
        Get representations based on a model, inputs and potentially targets.

        Args:
            model (RetainModel): A Retain model.
            inputs (th.Tensor): Input data.
            target (th.Tensor): Targets. Default to ``None``

        Returns:
            th.Tensor: attributions.
        """
        score = th.zeros(inputs.shape)

        logit, alpha, beta = model(
            inputs, (th.ones((len(inputs),)) * inputs.shape[1]).long()
        )
        w_emb = model.embedding[1].weight

        for i in range(inputs.shape[2]):
            for t in range(inputs.shape[1]):
                imp = model.output(
                    beta[:, t, :] * w_emb[:, i].expand_as(beta[:, t, :])
                )
                if target is None:
                    score[:, t, i] = (
                        alpha[:, t, 0] * imp.mean(-1) * inputs[:, t, i]
                    )
                else:
                    score[:, t, i] = (
                        alpha[:, t, 0]
                        * imp[
                            th.arange(0, len(imp)).long(),
                            target[:, -1, ...].long(),
                        ]
                        * inputs[:, t, i]
                    )
        return score.detach().cpu()

    @staticmethod
    def _format_target(inputs: tuple, target: TargetType):
        """
        Convert target into a Tensor.

        Args:
            inputs (tuple): Input data.
            target (TargetType): The target.

        Returns:
            th.Tensor: Converted target.
        """
        assert target is not None, "target must be provided"

        if isinstance(target, int):
            target = th.Tensor([target] * len(inputs[0]))

        if isinstance(target, list):
            target = th.Tensor(target)

        assert isinstance(target, th.Tensor), "Unsupported target."

        return target
