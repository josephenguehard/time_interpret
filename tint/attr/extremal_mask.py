import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_baseline,
    _format_input,
    _format_output,
    _is_tuple,
    _validate_input,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Any, Callable, Tuple

from tint.utils import TensorDataset, _add_temporal_mask, default_collate
from .models import ExtremalMaskNet


class ExtremalMask(PerturbationAttribution):
    """
    Extremal masks method.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    Examples:
        >>> import torch as th
        >>> from tint.attr import ExtremalMask
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = ExtremalMask(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        trainer: Trainer = None,
        mask_net: ExtremalMaskNet = None,
        batch_size: int = 32,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Attribute method.

        Args:
            inputs (tensor or tuple of tensors):  Input for which occlusion
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                Baselines define reference value which replaces each
                feature when occluded.
                Baselines can be provided as:

                - a single tensor, if inputs is a single tensor, with
                  exactly the same dimensions as inputs or
                  broadcastable to match the dimensions of inputs

                - a single scalar, if inputs is a single tensor, which will
                  be broadcasted for each input value in input tensor.

                - a tuple of tensors or scalars, the baseline corresponding
                  to each tensor in the inputs' tuple can be:

                  - either a tensor with matching dimensions to
                    corresponding tensor in the inputs' tuple
                    or the first dimension is one and the remaining
                    dimensions match with the corresponding
                    input tensor.

                  - or a scalar, corresponding to a tensor in the
                    inputs' tuple. This scalar value is broadcasted
                    for corresponding input tensor.

                In the cases when `baselines` is not provided, we internally
                use zero scalar corresponding to each input tensor.
                Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                which difference is computed (for classification cases,
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
                correspond to the number of examples. For all other types,
                the given argument is used for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided.
                Default: None
            mask_net (BayesMaskNet): A Mask model. If ``None``, a default model
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
        inputs = _format_input(inputs)

        # Format and validate baselines
        baselines = _format_baseline(baselines, inputs)
        _validate_input(inputs, baselines)

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
        baseline = baselines[0]

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
            mask_net = ExtremalMaskNet(forward_func=self.forward_func)

        # Init model
        mask_net.net.init(input_size=data.shape, batch_size=batch_size)

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(
                *(data, data, baseline, target, *additional_forward_args)
                if additional_forward_args is not None
                else (data, data, baseline, target, None)
            ),
            batch_size=batch_size,
            collate_fn=default_collate,
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        # Set model to eval mode
        mask_net.eval()

        # Get attributions as mask representation
        attributions = mask_net.net.representation(data, baseline)

        # Reshape representation if temporal attributions
        if return_temporal_attributions:
            attributions = attributions.reshape(
                (-1, data.shape[1]) + data.shape[1:]
            )

        # Reshape as a tuple
        attributions = (attributions,)

        # Format attributions and return
        return _format_output(is_inputs_tuple, attributions)
