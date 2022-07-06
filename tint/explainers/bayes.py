import torch as th
import torch.nn as nn

from captum.attr import Lime, KernelShap
from captum._utils.models import Model

from pytorch_lightning.trainer import Trainer
from sklearn.linear_model import BayesianRidge
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Union

from tint.models import Net
from tint.models.layers import BayesLinear


class BayesLime(Lime):
    """
    Bayesian version of Lime.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://arxiv.org/pdf/2008.05030
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(
            forward_func=forward_func,
            interpretable_model=BayesLinear(),
            similarity_func=None,
            perturb_func=None,
        )


class BayesShap(KernelShap):
    """
    Bayesian version of KernelShap.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        https://arxiv.org/pdf/2008.05030
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

        self.interpretable_model = BayesLinear()


class PytorchBayesModel(Model):
    def __init__(self, prior_sigma: float = 0.1, *args, **kwargs):
        super().__init__()
        self.prior_sigma = prior_sigma

        self.model = None
        self.trainer = Trainer(*args, **kwargs)

    def fit(
        self, train_data: DataLoader, **kwargs
    ) -> Optional[Dict[str, Union[int, float, th.Tensor]]]:
        layer = BayesLinear(
            prior_sigma=self.prior_sigma,
            in_features=next(iter(train_data)).shape[-1],
            out_features=1,
            bias=True,
        )

        self.model = Net(
            [layer],
            loss=lambda x, y: nn.MSELoss(x, y) + layer.loss(),
        )

    def representation(self) -> th.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        pass


class SklearnBayesModel(Model):
    def __init__(self):
        super().__init__()

    def fit(
        self, train_data: DataLoader, **kwargs
    ) -> Optional[Dict[str, Union[int, float, th.Tensor]]]:
        pass

    def representation(self) -> th.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        pass
