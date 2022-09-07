from captum.attr import Lime, KernelShap
from captum._utils.models import Model
from typing import Callable

from .models import BLRRidge


class BayesLime(Lime):
    """
    Bayesian version of Lime.

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
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model or BLRRidge(),
            similarity_func=None,
            perturb_func=None,
        )


class BayesShap(KernelShap):
    """
    Bayesian version of KernelShap.

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
        >>> from tint.attr import BayesShap
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = BayesShap(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
    ) -> None:
        super().__init__(forward_func=forward_func)

        self.interpretable_model = interpretable_model or BLRRidge()
