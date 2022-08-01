from captum.attr import KernelShap, Lime
from captum._utils.models.model import Model

from sklearn.neighbors import LocalOutlierFactor
from torch import Tensor
from typing import Callable, Optional

EPS = 1e-5


class LOF:
    def __init__(
        self,
        embeddings: Tensor,
        n_neighbors: int = 20,
        **kwargs,
    ):
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            **kwargs,
        )
        self.lof.fit(X=embeddings.reshape(len(embeddings), -1).numpy())

        self._similarity_func = None

    def lof_similarity_func(
        self,
        original_inp: Tensor,
        perturbed_inp: Tensor,
        interpretable_sample: Tensor,
        **kwargs,
    ):
        assert isinstance(
            original_inp, Tensor
        ), "Only one input is accepted with this method."
        score = self.lof.score_samples(
            perturbed_inp.reshape(len(perturbed_inp), -1).numpy()
        )
        train_scores = self.lof.negative_outlier_factor_
        score = (train_scores.max() - train_scores.min()) / (
            train_scores.max() - score
        )
        return self._similarity_func(
            original_inp,
            perturbed_inp,
            interpretable_sample,
            **kwargs,
        ) * max(score.item(), EPS)


class LOFLime(Lime, LOF):
    def __init__(
        self,
        forward_func: Callable,
        embeddings: Tensor,
        n_neighbors: int = 20,
        interpretable_model: Optional[Model] = None,
        similarity_func: Optional[Callable] = None,
        perturb_func: Optional[Callable] = None,
        **kwargs,
    ):
        Lime.__init__(
            self,
            forward_func=forward_func,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )
        LOF.__init__(
            self,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
            **kwargs,
        )

        # Replace original similarity_func with the custom one
        self._similarity_func = self.similarity_func
        self.similarity_func = self.lof_similarity_func


class LOFKernelShap(KernelShap, LOF):
    def __init__(
        self,
        forward_func: Callable,
        embeddings: Tensor,
        n_neighbors: int = 20,
        **kwargs,
    ):
        KernelShap.__init__(self, forward_func=forward_func)
        LOF.__init__(
            self,
            embeddings=embeddings,
            n_neighbors=n_neighbors,
            **kwargs,
        )

        # Replace original similarity_func with the custom one
        self._similarity_func = self.similarity_func
        self.similarity_func = self.lof_similarity_func
