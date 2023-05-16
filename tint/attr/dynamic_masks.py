import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_inputs,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Any, Callable, Tuple

from tint.utils import TensorDataset, _add_temporal_mask, default_collate
from .models import MaskNet


class DynaMask(PerturbationAttribution):
    """
    Dynamic masks.

    This method aims to explain time series data, by learning a mask
    representing features importance. This method was inspired from
    Fong et al., and can be used in "preservation game" mode: trying to keep
    the closest predictions, compared with unperturebed data, with the
    minimal number of features, or in "deletion game" mode, trying to get the
    furthest predictions by removing the minimal number of features.

    This implementation batchify the original method by leanrning in parallel
    multiple inputs and multiple ``keep_ratio`` (called ``mask_group`` in the
    original implementation.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://arxiv.org/pdf/2106.05303
        https://arxiv.org/pdf/1910.08485

    Examples:
        >>> import torch as th
        >>> from tint.attr import DynaMask
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = DynaMask(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Any = None,
        trainer: Trainer = None,
        mask_net: MaskNet = None,
        batch_size: int = 32,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        return_best_ratio: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Attribute method.

        Args:
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            additional_forward_args (any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. It will be
                repeated for each of `n_steps` along the integrated
                path. For all other types, the given argument is used
                for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided.
                Default: None
            mask_net (MaskNet): A Mask model. If ``None``, a default model
                will be provided.
                Default: None
            batch_size (int): Batch size for Mask training.
                Default: 32
            temporal_additional_forward_args (tuple): Set each
                additional forward arg which is temporal.
                Only used with return_temporal_attributions.
                Default: None
            return_temporal_attributions (bool): Whether to return
                attributions for all times or not.
                Default: False
            return_best_ratio (bool): Whether to return the best keep_ratio
                or not.
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

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)
        else:
            trainer = copy.deepcopy(trainer)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"
        data = inputs[0]

        # If return temporal attr, we expand the input data
        # and multiply it with a lower triangular mask
        if return_temporal_attributions:
            data, additional_forward_args, _ = _add_temporal_mask(
                inputs=data,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )

        # Init MaskNet if not provided
        if mask_net is None:
            mask_net = MaskNet(forward_func=self.forward_func)
        else:
            mask_net = copy.deepcopy(mask_net)

        # Init model
        mask_net.net.init(
            shape=data.shape,
            n_epochs=trainer.max_epochs or trainer.max_steps,
            batch_size=batch_size,
        )

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(
                *(data, data, *additional_forward_args)
                if additional_forward_args is not None
                else (data, data, None)
            ),
            batch_size=batch_size,
            collate_fn=default_collate,
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        # Set model to eval mode
        mask_net.eval()

        # Get attributions as mask representation
        attributions, best_ratio = self.representation(
            mask_net=mask_net,
            trainer=trainer,
            dataloader=dataloader,
        )

        # Reshape representation if temporal attributions
        if return_temporal_attributions:
            attributions = attributions.reshape(
                (-1, data.shape[1]) + data.shape[1:]
            )

        # Reshape as a tuple
        attributions = (attributions,)

        # Format attributions and return
        if return_best_ratio:
            return _format_output(is_inputs_tuple, attributions), best_ratio
        return _format_output(is_inputs_tuple, attributions)

    @staticmethod
    def representation(
        mask_net: MaskNet, trainer: Trainer, dataloader: DataLoader
    ):
        mask = (
            1.0 - mask_net.net.mask
            if mask_net.net.deletion_mode
            else mask_net.net.mask
        )

        # Get the loss without reduction
        pred = trainer.predict(mask_net, dataloaders=dataloader)
        _loss = mask_net._loss
        _loss.reduction = "none"
        loss = _loss(
            th.cat([x[0] for x in pred]), th.cat([x[1] for x in pred])
        )

        # Average the loss over each keep_ratio subset
        if len(loss.shape) > 1:
            loss = loss.sum(tuple(range(1, len(loss.shape))))
        loss = loss.reshape(
            len(mask_net.net.keep_ratio),
            len(loss) // len(mask_net.net.keep_ratio),
        )
        loss = loss.sum(-1)

        # Get the minimum loss
        i = loss.argmin().item()
        length = len(mask) // len(mask_net.net.keep_ratio)

        # Return the mask subset given the minimum loss
        return (
            mask.detach().cpu()[i * length : (i + 1) * length],
            mask_net.net.keep_ratio[i],
        )
