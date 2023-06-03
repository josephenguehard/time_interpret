import torch

from captum.attr import Lime, KernelShap, LimeBase
from captum.attr._core.lime import construct_feature_mask
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import _format_input_baseline
from captum.log import log_usage
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


class _BayesLime(Lime):
    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
        similarity_func: Optional[Callable] = None,
        perturb_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
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


class BayesLime(_BayesLime):
    """
    Bayesian version of Lime.

    This method replace the linear regression of the original Lime with a
    bayesian linear regression, allowing to model uncertainty in
    explainability.

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
        percent (int): Percentage for the credible intervals. Must be between
                0 and 100. Only used when no custom interpretable model is
                passed. Otherwise, you must specify the percentage for
                credible interval in the model definition:

                >>> from tint.attr.models import BLRRegression
                >>> model = BLRRegression(percent=90)
                >>> explainer = BayesLime(mlp, interpretable_model=model)

                Default: 95

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
        >>> attr, credible_int = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
        similarity_func: Callable = None,
        perturb_func: Callable = None,
        percent: int = 95,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model
            or BLRRidge(percent=percent),
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )

    @log_usage()
    def attribute(  # type: ignore
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
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. Note that provided similarity and perturbation functions will be
        provided each example separately (first dimension = 1) in this case.
        If forward_fn returns a scalar per batch (e.g. loss), attributions will
        still be computed using a single interpretable model for the full batch.
        In this case, similarity and perturbation functions will be provided the
        same original input containing the full batch.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

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
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
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
            additional_forward_args (Any, optional): If the forward function
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
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretale feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False

        Returns:
            2-element tuple of **attributions**, **credible_intervals**:
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

        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining Lime interpreter
            >>> lime = Lime(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = lime.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes interpretable model and returning attributions
            >>> # matching input shape.
            >>> attr = lime.attribute(input, target=1, feature_mask=feature_mask)
        """
        return super().attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )


class BayesKernelShap(KernelShap, _BayesLime):
    """
    Bayesian version of KernelShap.

    This method replace the linear regression of the original KernelShap with
    a bayesian linear regression, allowing to model uncertainty in
    explainability.

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

        percent (int): Percentage for the credible intervals. Must be between
                0 and 100. Only used when no custom interpretable model is
                passed. Otherwise, you must specify the percentage for
                credible interval in the model definition:

                >>> from tint.attr.models import BLRRegression
                >>> model = BLRRegression(percent=90)
                >>> explainer = BayesKernelShap(mlp, interpretable_model=model)

                Default: 95

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
        >>> attr, credible_int = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
        percent: int = 95,
    ) -> None:
        super().__init__(forward_func=forward_func)

        self.interpretable_model = interpretable_model or BLRRidge(
            percent=percent
        )

    @log_usage()
    def attribute(  # type: ignore
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
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model based on KernelSHAP and returning a
        representation of the interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME / KernelShap
        is generally used for sample-based interpretability, training a separate
        interpretable model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. Note that provided similarity and perturbation functions will be
        provided each example separately (first dimension = 1) in this case.
        If forward_fn returns a scalar per batch (e.g. loss), attributions will
        still be computed using a single interpretable model for the full batch.
        In this case, similarity and perturbation functions will be provided the
        same original input containing the full batch.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which KernelShap
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

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
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
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
            additional_forward_args (Any, optional): If the forward function
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
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretable feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpretable model, with length
                        num_interp_features.
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False

        Returns:
            2-element tuple of **attributions**, **credible_intervals**:
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

        Examples::
            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining KernelShap interpreter
            >>> ks = KernelShap(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = ks.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes KernelSHAP attributions with feature mask.
            >>> attr = ks.attribute(input, target=1, feature_mask=feature_mask)
        """
        return super().attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )
