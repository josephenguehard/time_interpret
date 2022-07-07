from captum.attr import Lime, KernelShap
from captum._utils.models import Model
from typing import Callable

from .models import SkLearnBayesianRidge


class BayesLime(Lime):
    """
    Bayesian version of Lime.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        interpretable_model (Model): Model object to train interpretable model.
            Default to ``None``

    References:
        https://arxiv.org/pdf/2008.05030
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
    ) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model or SkLearnBayesianRidge(),
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
            Default to ``None``

    References:
        https://arxiv.org/pdf/2008.05030
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model = None,
    ) -> None:
        super().__init__(forward_func=forward_func)

        self.interpretable_model = (
            interpretable_model or SkLearnBayesianRidge()
        )
