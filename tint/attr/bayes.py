import torch

from captum.attr import Lime, KernelShap, LimeBase
from captum.attr._core.lime import construct_feature_mask
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import _format_input_baseline
from captum._utils.models import Model
from captum._utils.common import (
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union
import warnings

from .models import BLRRidge


class _Lime(Lime):
    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
        similarity_func: Optional[Callable] = None,
        perturb_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model or BLRRidge(),
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )

    def _attribute_kwargs(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. "
            )

        coefs: Tensor
        if bsz > 1:
            test_output = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            if (
                isinstance(test_output, Tensor)
                and torch.numel(test_output) > 1
            ):
                if torch.numel(test_output) == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a time."
                    )
                    output_list = []
                    creds_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                    ):
                        coefs, creds = LimeBase.attribute.__wrapped__(
                            self,
                            inputs=curr_inps
                            if is_inputs_tuple
                            else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=curr_baselines
                            if is_inputs_tuple
                            else curr_baselines[0],
                            feature_mask=curr_feature_mask
                            if is_inputs_tuple
                            else curr_feature_mask[0],
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        if return_input_shape:
                            output_list.append(
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    coefs,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                            )
                            creds_list.append(
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    creds,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                            )
                        else:
                            output_list.append(coefs.reshape(1, -1))  # type: ignore
                            creds_list.append(creds.reshape(1, -1))  # type: ignore

                    return _reduce_list(output_list), _reduce_list(creds_list)
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        coefs, creds = LimeBase.attribute.__wrapped__(
            self,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )
        if return_input_shape:
            return self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                coefs,
                num_interp_features,
                is_inputs_tuple,
            ), self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                creds,
                num_interp_features,
                is_inputs_tuple,
            )
        else:
            return coefs, creds


class BayesLime(_Lime):
    """
    Bayesian version of Lime.

    This method replace the linear regression of the original Lime with a
    bayesian linear regression, allowing to model uncertainty in
    explainability.

    Its attribution method therefore returns a tuple of two tensors:
    - **attributions** (*tensor* or tuple of *tensors*):
        The attributions with respect to each input feature.
        If return_input_shape = True, attributions will be
        the same size as the provided inputs, with each value
        providing the coefficient of the corresponding
        interpretale feature.
        If return_input_shape is False, a 1D
        tensor is returned, containing only the coefficients
        of the trained interpreatable models, with length
        num_interp_features.
    - **credible_intervals** (*tensor* or tuple of *tensors*):
        The credible intervals associated with each attribution.
        If return_input_shape = True, credible intervals will be
        the same size as the provided inputs, with each value
        providing the coefficient of the corresponding
        interpretale feature.
        If return_input_shape is False, a 1D
        tensor is returned, containing only the credible intervals
        of the trained interpreatable models, with length
        num_interp_features.

    Args:
        forward_func (callable): The forward function of the model or any
                modification of it.
        interpretable_model (Model): Model object to train interpretable model.

                This argument is optional and defaults to SkLearnBayesianRidge(),
                which is a wrapper around the Bayesian Ridge in SkLearn.
                This requires having sklearn version >= 0.23 available.

                Other predefined interpretable linear models are provided in
                tint.attr.models.bayes_linear.

                Alternatively, a custom model object must provide a `fit` method to
                train the model, given a dataloader, with batches containing
                three tensors:

                - interpretable_inputs: Tensor
                  [2D num_samples x num_interp_features],
                - expected_outputs: Tensor [1D num_samples],
                - weights: Tensor [1D num_samples]

                The model object must also provide a `representation` method to
                access the appropriate coefficients or representation of the
                interpretable model after fitting.

                Note that calling fit multiple times should retrain the
                interpretable model, each attribution call reuses
                the same given interpretable model object.

                Default: None
        similarity_func (Callable, optional): Function which takes a single sample
                along with its corresponding interpretable representation
                and returns the weight of the interpretable sample for
                training the interpretable model.
                This is often referred to as a similarity kernel.

                This argument is optional and defaults to a function which
                applies an exponential kernel to the cosine distance between
                the original input and perturbed input, with a kernel width
                of 1.0.

                A similarity function applying an exponential
                kernel to cosine / euclidean distances can be constructed
                using the provided get_exp_kernel_similarity_function in
                captum.attr._core.lime.

                Alternately, a custom callable can also be provided.
                The expected signature of this callable is:

                >>> def similarity_func(
                >>>    original_input: Tensor or tuple[Tensor, ...],
                >>>    perturbed_input: Tensor or tuple[Tensor, ...],
                >>>    perturbed_interpretable_input:
                >>>        Tensor [2D 1 x num_interp_features],
                >>>    **kwargs: Any
                >>> ) -> float or Tensor containing float scalar

                perturbed_input and original_input will be the same type and
                contain tensors of the same shape, with original_input
                being the same as the input provided when calling attribute.

                kwargs includes baselines, feature_mask, num_interp_features
                (integer, determined from feature mask).
        perturb_func (Callable, optional): Function which returns a single
                sampled input, which is a binary vector of length
                num_interp_features, or a generator of such tensors.

                This function is optional, the default function returns
                a binary vector where each element is selected
                independently and uniformly at random. Custom
                logic for selecting sampled binary vectors can
                be implemented by providing a function with the
                following expected signature:

                >>> perturb_func(
                >>>    original_input: Tensor or tuple[Tensor, ...],
                >>>    **kwargs: Any
                >>> ) -> Tensor [Binary 2D Tensor 1 x num_interp_features]
                >>>  or generator yielding such tensors

                kwargs includes baselines, feature_mask, num_interp_features
                (integer, determined from feature mask).

    References:
        https://arxiv.org/pdf/2008.05030

    Examples:
        >>> import torch as th
        >>> from tint.attr import BayesLime
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = BayesLime(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
        similarity_func: Callable = None,
        perturb_func: Callable = None,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model or BLRRidge(),
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )


class BayesKernelShap(KernelShap, _Lime):
    """
    Bayesian version of KernelShap.

    This method replace the linear regression of the original KernelShap with
    a bayesian linear regression, allowing to model uncertainty in
    explainability.

    Its attribution method therefore returns a tuple of two tensors:
    - **attributions** (*tensor* or tuple of *tensors*):
        The attributions with respect to each input feature.
        If return_input_shape = True, attributions will be
        the same size as the provided inputs, with each value
        providing the coefficient of the corresponding
        interpretale feature.
        If return_input_shape is False, a 1D
        tensor is returned, containing only the coefficients
        of the trained interpreatable models, with length
        num_interp_features.
    - **credible_intervals** (*tensor* or tuple of *tensors*):
        The credible intervals associated with each attribution.
        If return_input_shape = True, credible intervals will be
        the same size as the provided inputs, with each value
        providing the coefficient of the corresponding
        interpretale feature.
        If return_input_shape is False, a 1D
        tensor is returned, containing only the credible intervals
        of the trained interpreatable models, with length
        num_interp_features.

    Args:
        forward_func (callable): The forward function of the model or any
                modification of it.
        interpretable_model (Model): Model object to train interpretable model.

                This argument is optional and defaults to SkLearnBayesianRidge(),
                which is a wrapper around the Bayesian Ridge in SkLearn.
                This requires having sklearn version >= 0.23 available.

                Other predefined interpretable linear models are provided in
                tint.attr.models.bayes_linear.

                Alternatively, a custom model object must provide a `fit` method to
                train the model, given a dataloader, with batches containing
                three tensors:

                - interpretable_inputs: Tensor
                  [2D num_samples x num_interp_features],
                - expected_outputs: Tensor [1D num_samples],
                - weights: Tensor [1D num_samples]

                The model object must also provide a `representation` method to
                access the appropriate coefficients or representation of the
                interpretable model after fitting.

                Note that calling fit multiple times should retrain the
                interpretable model, each attribution call reuses
                the same given interpretable model object.

                Default: None

    References:
        https://arxiv.org/pdf/2008.05030

    Examples:
        >>> import torch as th
        >>> from tint.attr import BayesKernelShap
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = BayesKernelShap(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
    ) -> None:
        super().__init__(forward_func=forward_func)

        self.interpretable_model = interpretable_model or BLRRidge()
