from captum.attr import KernelShap, Lime
from captum._utils.models.model import Model
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.neighbors import LocalOutlierFactor
from torch import Tensor
from typing import Callable, Optional

EPS = 1e-5


class LOF:
    """
    Local Outlier Factor Lime.

    Args:
        embeddings (Tensor): Tensor of embeddings to compute the LOF.
        n_neighbors (int): Number of neighbors to use by default.
            Default to 20
    """

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
        self.lof.fit(X=embeddings.reshape(len(embeddings), -1).cpu().numpy())

        self._similarity_func = None

    def lof_similarity_func(
        self,
        original_inp: TensorOrTupleOfTensorsGeneric,
        perturbed_inp: TensorOrTupleOfTensorsGeneric,
        interpretable_sample: TensorOrTupleOfTensorsGeneric,
        **kwargs,
    ):
        # Only use the first input if tuple
        # Lof only accepts one input
        pert_inp = perturbed_inp
        if isinstance(perturbed_inp, tuple):
            assert (
                len(perturbed_inp) == 1
            ), "Only one input is accepted with this method."
            pert_inp = perturbed_inp[0]

        score = -self.lof.score_samples(
            pert_inp.reshape(len(pert_inp), -1).cpu().numpy()
        )
        score = 1 / score.clip(min=1)
        return self._similarity_func(
            original_inp,
            perturbed_inp,
            interpretable_sample,
            **kwargs,
        ) * max(score.mean().item(), EPS)


class LofLime(Lime, LOF):
    r"""
    Local Outlier Factor Lime.

    This method compute a Local Outlier Factor score for every perturbed data.
    This score is then used to update the weight given by the similarity
    function:

    .. math::
        new_weight(x) = similarity(x) * \frac{-1}{lof_score(x)}

    If the perturbed data is considered more out of sample, the weight of
    this data will be reduced.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it.
        embeddings (Tensor): Tensor of embeddings to compute the LOF.
        n_neighbors (int): Number of neighbors to use by default.
            Default to 20
        interpretable_model (optional, Model): Model object to train
            interpretable model.

            This argument is optional and defaults to SkLearnLasso(alpha=0.01),
            which is a wrapper around the Lasso linear model in SkLearn.
            This requires having sklearn version >= 0.23 available.

            Other predefined interpretable linear models are provided in
            captum._utils.models.linear_model.

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
        similarity_func (optional, callable): Function which takes a single sample
            along with its corresponding interpretable representation
            and returns the weight of the interpretable sample for
            training the interpretable model.
            This is often referred to as a similarity kernel.

            This argument is optional and defaults to a function which
            applies an exponential kernel to the consine distance between
            the original input and perturbed input, with a kernel width
            of 1.0.

            A similarity function applying an exponential
            kernel to cosine / euclidean distances can be constructed
            using the provided get_exp_kernel_similarity_function in
            captum.attr._core.lime.

            Alternately, a custom callable can also be provided.
            The expected signature of this callable is:

            >>> def similarity_func(
            >>>    original_input: Tensor or tuple of Tensors,
            >>>    perturbed_input: Tensor or tuple of Tensors,
            >>>    perturbed_interpretable_input:
            >>>        Tensor [2D 1 x num_interp_features],
            >>>    **kwargs: Any
            >>> ) -> float or Tensor containing float scalar

            perturbed_input and original_input will be the same type and
            contain tensors of the same shape, with original_input
            being the same as the input provided when calling attribute.

            kwargs includes baselines, feature_mask, num_interp_features
            (integer, determined from feature mask).
        perturb_func (optional, callable): Function which returns a single
            sampled input, which is a binary vector of length
            num_interp_features, or a generator of such tensors.

            This function is optional, the default function returns
            a binary vector where each element is selected
            independently and uniformly at random. Custom
            logic for selecting sampled binary vectors can
            be implemented by providing a function with the
            following expected signature:

            >>> perturb_func(
            >>>    original_input: Tensor or tuple of Tensors,
            >>>    **kwargs: Any
            >>> ) -> Tensor [Binary 2D Tensor 1 x num_interp_features]
            >>>  or generator yielding such tensors

            kwargs includes baselines, feature_mask, num_interp_features
            (integer, determined from feature mask).

    References:
        `Time Interpret: a Unified Model Interpretability Library for Time Series <https://arxiv.org/abs/2306.02968>`_

    Examples:
        >>> import torch as th
        >>> from tint.attr import LofLime
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = LofLime(mlp, data, n_neighbors=2)
        >>> attr = explainer.attribute(inputs, target=0)
    """

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


class LofKernelShap(KernelShap, LOF):
    r"""
    Local Outlier Factor Kernel Shap.

    This method compute a Local Outlier Factor score for every perturbed data.
    This score is then used to update the weight given by the similarity
    function:

    .. math::
        new_weight(x) = similarity(x) * \frac{-1}{lof_score(x)}

    If the perturbed data is considered more out of sample, the weight of
    this data will be reduced.

    Args:
        forward_func (Callable): The forward function of the model or any
            modification of it.
        embeddings (Tensor): Tensor of embeddings to compute the LOF.
        n_neighbors (int): Number of neighbors to use by default.
            Default to 20

    References:
        `Time Interpret: a Unified Model Interpretability Library for Time Series <https://arxiv.org/abs/2306.02968>`_

    Examples:
        >>> import torch as th
        >>> from tint.attr import LofKernelShap
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = LofKernelShap(mlp, data, n_neighbors=2)
        >>> attr = explainer.attribute(inputs, target=0)
    """

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
