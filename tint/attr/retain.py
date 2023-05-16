import copy

import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_inputs,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric, TargetType

from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Union

from tint.utils import _add_temporal_mask
from .models import Retain as RetainModel, RetainNet


class Retain(PerturbationAttribution):
    """
    Retain explainer method.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it.
        retain (RetainNet): A Retain network as a Pytorch Lightning
            module. If ``None``, a default Retain Net will be created.
            Default to ``None``
        datamodule (LightningDataModule): A Pytorch Lightning data
            module which will be used to train the RetainNet.
            Either a datamodule or features must be provided, they cannot be
            None together. Default to ``None``
        features (Tensor): A tensor of features which will be used to train
            the RetainNet. Either a datamodule or features must be provided,
            they cannot be None together. If both are provided, features is
            ignored. Default to ``None``

    References:
        https://arxiv.org/pdf/1608.05745

    Examples:
        >>> import torch as th
        >>> from tint.attr import Retain
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> labels = th.randint(2, (32, 7))
        <BLANKLINE>
        >>> explainer = Retain(features=data, labels=labels)
        >>> attr = explainer.attribute(inputs, target=th.randint(2, (8, 7))))
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
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                which gradients are computed (for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example,
                no target index is necessary.
                For general 2D outputs, targets can be either:

                - a single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                  the number of examples in inputs (dim 0). Each integer
                  is applied as the target for the corresponding example.

                For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                  elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                  examples in inputs (dim 0), and each tuple containing
                  #output_dims - 1 elements. Each tuple is applied as the
                  target for the corresponding example.

                Default: None
            return_temporal_attributions (bool): Whether to return
                attributions for all times or not.
                Default: False

        Returns:
            - **attributions** (*tensor* or tuple of *tensors*):
                The attributions with respect to each input feature.
                Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_inputs(inputs)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"

        # Make target a tensor
        target = self._format_target(inputs, target)

        # Get data as only value in inputs
        data = inputs[0]

        # Set generator to device
        self.forward_func.to(data.device)

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
                inputs=data,
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
        score = th.zeros(inputs.shape).to(inputs.device)

        logit, alpha, beta = self.forward_func(
            inputs,
            (th.ones((len(inputs),)) * inputs.shape[1]).int(),
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
    def _format_target(
        inputs: tuple, target: TargetType
    ) -> Union[None, th.Tensor]:
        """
        Convert target into a Tensor.

        Args:
            inputs (tuple): Input data.
            target (TargetType): The target.

        Returns:
            th.Tensor: Converted target.
        """
        if target is None:
            return None

        if isinstance(target, int):
            target = th.Tensor([target] * len(inputs[0]))

        if isinstance(target, list):
            target = th.Tensor(target)

        assert isinstance(target, th.Tensor), "Unsupported target."

        return target
