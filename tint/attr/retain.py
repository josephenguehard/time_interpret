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

from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

from tint.utils import _add_temporal_mask
from .models import Retain as RetainModel, RetainNet


class Retain(PerturbationAttribution):
    """
    Retain explainer method.

    References:
        https://arxiv.org/pdf/1608.05745
    """

    def __init__(
        self,
        forward_func: Callable = None,
        retain: RetainNet = None,
        datamodule: LightningDataModule = None,
        features: th.Tensor = None,
        labels: th.Tensor = None,
        trainer: Trainer = None,
        batch_size: int = 32,
    ) -> None:
        # If forward_func is not provided,
        # train retain model
        if forward_func is None:

            # Create dataloader if not provided
            dataloader = None
            if datamodule is None:
                assert (
                    features is not None
                ), "You must provide either a datamodule or features"
                assert (
                    labels is not None
                ), "You must provide either a datamodule or labels"

                dataloader = DataLoader(
                    TensorDataset(features, labels),
                    batch_size=batch_size,
                )

            # Init trainer if not provided
            if trainer is None:
                trainer = Trainer(max_epochs=100)
            else:
                trainer = copy.deepcopy(trainer)

            # Create retain if not provided
            if retain is None:
                retain = RetainNet(loss="cross_entropy")
            else:
                # LazyLinear cannot be deep copied
                pass

            # Train retain
            trainer.fit(
                retain, train_dataloaders=dataloader, datamodule=datamodule
            )

            # Set to eval mode
            retain.eval()

            # Extract forward_func from model
            forward_func = retain.net

        super().__init__(forward_func=forward_func)

        assert isinstance(
            forward_func, RetainModel
        ), "Only a Retain model can be used here."

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        return_temporal_attributions: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            target (int, tuple, tensor, list): Output indices. Default to
                ``None``
            return_temporal_attributions (bool): Whether to return
                attributions for all times or not. Default to ``False``

        Returns:
            (th.Tensor, tuple): Attributions.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_input(inputs)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"

        # Make target a tensor
        target = self._format_target(inputs, target)

        # Get data as only value in inputs
        data = inputs[0]

        # If return temporal attr, we expand the input data
        # and multiply it with a lower triangular mask
        if return_temporal_attributions:
            data, _, target = _add_temporal_mask(
                inputs=data,
                target=target,
                temporal_target=self.forward_func.temporal_labels,
            )

        # Get attributions
        attributions = (
            self.representation(
                inputs=inputs[0],
                target=target,
            ),
        )

        # Reshape attributions if temporal attributions
        if return_temporal_attributions:
            attributions = (
                attributions[0].reshape((-1, data.shape[1]) + data.shape[1:]),
            )

        return _format_output(is_inputs_tuple, attributions)

    def representation(
        self,
        inputs: th.Tensor,
        target: th.Tensor = None,
    ):
        """
        Get representations based on a model, inputs and potentially targets.

        Args:
            inputs (th.Tensor): Input data.
            target (th.Tensor): Targets. Default to ``None``

        Returns:
            th.Tensor: attributions.
        """
        score = th.zeros(inputs.shape)

        logit, alpha, beta = self.forward_func(
            inputs, (th.ones((len(inputs),)) * inputs.shape[1]).long()
        )
        w_emb = self.forward_func.embedding[1].weight

        if target is not None and self.forward_func.temporal_labels:
            target = target[:, -1, ...]

        for i in range(inputs.shape[2]):
            for t in range(inputs.shape[1]):
                imp = self.forward_func.output(
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
                            target,
                        ]
                        * inputs[:, t, i]
                    )
        return score.detach().cpu()

    @staticmethod
    def _format_target(inputs: tuple, target: TargetType) -> th.Tensor:
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
