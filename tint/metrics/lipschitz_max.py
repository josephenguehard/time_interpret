from copy import deepcopy
from inspect import signature
from typing import Any, Callable, cast, Tuple, Union

import torch
from captum._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_target,
    _format_baseline,
    _format_tensor_into_tuples,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from captum.metrics._utils.batching import _divide_and_aggregate_metrics
from torch import Tensor

from tint.utils import default_perturb_func


@log_usage()
def lipschitz_max(
    explanation_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 10,
    norm_ord: str = "fro",
    max_examples_per_batch: int = None,
    **kwargs: Any,
) -> Tensor:
    r"""
    Lipschitz Max as a stability metric.

    Args:

        explanation_func (callable):
            This function can be the `attribute` method of an
            attribution algorithm or any other explanation method
            that returns the explanations.

        inputs (tensor or tuple of tensors):  Input for which
            explanations are computed. If `explanation_func` takes a
            single tensor as input, a single input tensor should
            be provided.
            If `explanation_func` takes multiple tensors as input, a tuple
            of the input tensors should be provided. It is assumed
            that for all given input tensors, dimension 0 corresponds
            to the number of examples (aka batch size), and if
            multiple input tensors are provided, the examples must
            be aligned appropriately.

        perturb_func (callable):
            The perturbation function of model inputs. This function takes
            model inputs and optionally `perturb_radius` if
            the function takes more than one argument and returns
            perturbed inputs.

            If there are more than one inputs passed to sensitivity function those
            will be passed to `perturb_func` as tuples in the same order as they
            are passed to sensitivity function.

            It is important to note that for performance reasons `perturb_func`
            isn't called for each example individually but on a batch of
            input examples that are repeated `max_examples_per_batch / batch_size`
            times within the batch.
            Default: default_perturb_func

        perturb_radius (float, optional): The epsilon radius used for sampling.
            In the `default_perturb_func` it is used as the radius of
            the L-Infinity ball. In a general case it can serve as a radius of
            any L_p nom.
            This argument is passed to `perturb_func` if it takes more than
            one argument.
            Default: 0.02

        n_perturb_samples (int, optional): The number of times input tensors
            are perturbed. Each input example in the inputs tensor is
            expanded `n_perturb_samples` times before calling
            `perturb_func` function.
            Default: 10

        norm_ord (int, float, inf, -inf, 'fro', 'nuc', optional): The type of norm
            that is used to compute the
            norm of the sensitivity matrix which is defined as the difference
            between the explanation function at its input and perturbed input.
            Default: 'fro'

        max_examples_per_batch (int, optional): The number of maximum input
            examples that are processed together. In case the number of
            examples (`input batch size * n_perturb_samples`) exceeds
            `max_examples_per_batch`, they will be sliced
            into batches of `max_examples_per_batch` examples and processed
            in a sequential order. If `max_examples_per_batch` is None, all
            examples are processed together. `max_examples_per_batch` should
            at least be equal `input batch size` and at most
            `input batch size * n_perturb_samples`.
            Default: None

        **kwargs (Any, optional): Contains a list of arguments that are passed
            to `explanation_func` explanation function which in some cases
            could be the `attribute` function of an attribution algorithm.
            Any additional arguments that need be passed to the explanation
            function should be included here.
            For instance, such arguments include:
            `additional_forward_args`, `baselines` and `target`.

    Returns:

        sensitivities (tensor): A tensor of scalar sensitivity scores per
            input example. The first dimension is equal to the
            number of examples in the input batch and the second
            dimension is one. Returned sensitivities are normalized by
            the magnitudes of the input explanations.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes sensitivity score for saliency maps of class 3
        >>> sens = lipschitz_max(saliency.attribute, input, target = 3)

    """

    def _generate_perturbations(
        current_n_perturb_samples: int,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For perfomance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples` repeated instances
        per example.
        """
        inputs_expanded: Union[Tensor, Tuple[Tensor, ...]] = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )
        if len(inputs_expanded) == 1:
            inputs_expanded = inputs_expanded[0]

        return (
            perturb_func(inputs_expanded, perturb_radius)
            if len(signature(perturb_func).parameters) > 1
            else perturb_func(inputs_expanded)
        )

    def max_values(input_tnsr: Tensor) -> Tensor:
        return torch.max(input_tnsr, dim=1).values  # type: ignore

    kwarg_expanded_for = None
    kwargs_copy: Any = None

    def _next_sensitivity_max(current_n_perturb_samples: int) -> Tensor:
        inputs_perturbed = _generate_perturbations(current_n_perturb_samples)

        # copy kwargs and update some of the arguments that need to be expanded
        nonlocal kwarg_expanded_for
        nonlocal kwargs_copy
        if (
            kwarg_expanded_for is None
            or kwarg_expanded_for != current_n_perturb_samples
        ):
            kwarg_expanded_for = current_n_perturb_samples
            kwargs_copy = deepcopy(kwargs)
            _expand_and_update_additional_forward_args(
                current_n_perturb_samples, kwargs_copy
            )
            _expand_and_update_target(current_n_perturb_samples, kwargs_copy)
            if "baselines" in kwargs:
                baselines = kwargs["baselines"]
                baselines = _format_baseline(
                    baselines, cast(Tuple[Tensor, ...], inputs)
                )
                if (
                    isinstance(baselines[0], Tensor)
                    and baselines[0].shape == inputs[0].shape
                ):
                    _expand_and_update_baselines(
                        cast(Tuple[Tensor, ...], inputs),
                        current_n_perturb_samples,
                        kwargs_copy,
                    )

        expl_perturbed_inputs = explanation_func(
            inputs_perturbed, **kwargs_copy
        )

        # tuplize `expl_perturbed_inputs` in case it is not
        expl_perturbed_inputs = _format_tensor_into_tuples(
            expl_perturbed_inputs
        )

        expl_inputs_expanded = tuple(
            expl_input.repeat_interleave(current_n_perturb_samples, dim=0)
            for expl_input in expl_inputs
        )

        sensitivities = torch.cat(
            [
                (expl_input - expl_perturbed).view(expl_perturbed.size(0), -1)
                for expl_perturbed, expl_input in zip(
                    expl_perturbed_inputs, expl_inputs_expanded
                )
            ],
            dim=1,
        )

        inputs_expanded: Union[Tensor, Tuple[Tensor, ...]] = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )

        # compute ||inputs - inputs_pert||
        inputs_diff = torch.cat(
            [
                (input - input_pert).view(input_pert.size(0), -1)
                for input, input_pert in zip(inputs_expanded, inputs_perturbed)
            ],
            dim=1,
        )

        inputs_diff_norm = torch.norm(
            inputs_diff,
            p=norm_ord,
            dim=1,
            keepdim=True,
        )

        inputs_diff_norm = torch.where(
            inputs_diff_norm == 0.0,
            torch.tensor(
                1.0,
                device=inputs_diff_norm.device,
                dtype=inputs_diff_norm.dtype,
            ),
            inputs_diff_norm,
        )

        # compute the norm for each input noisy example
        lipschitz = (
            torch.norm(sensitivities, p=norm_ord, dim=1, keepdim=True)
            / inputs_diff_norm
        )
        return max_values(lipschitz.view(bsz, -1))

    inputs = _format_tensor_into_tuples(inputs)  # type: ignore

    bsz = inputs[0].size(0)

    with torch.no_grad():
        expl_inputs = explanation_func(inputs, **kwargs)
        metrics_max = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_sensitivity_max,
            max_examples_per_batch=max_examples_per_batch,
            agg_func=torch.max,
        )
    return metrics_max
