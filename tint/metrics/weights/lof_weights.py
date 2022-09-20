import torch as th
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from sklearn.neighbors import LocalOutlierFactor


def lof_weights(
    data: TensorOrTupleOfTensorsGeneric,
    n_neighbors: int = 20,
    **kwargs,
):
    """
    Compute weights given original and perturbed inputs.

    Args:
        data (tensor or tuple of tensors): Data to fit the lof.
        n_neighbors (int, optional): Number of neighbors for the lof.
            Default: 20
        **kwargs: Additional arguments for the lof.

    Returns:
        Callable: A function to compute weights given original and
            perturbed inputs.
    """
    # Format data
    data = _format_tensor_into_tuples(data)

    # Def and fit lof
    lof_tpl = tuple(
        LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            **kwargs,
        )
        for _ in data
    )
    for x, lof in zip(data, lof_tpl):
        lof.fit(X=x.reshape(len(x), -1).cpu().numpy())

    def weights_fn(inputs, inputs_pert):
        # Compute lof scores
        score_tpl = tuple()
        for input_pert in inputs_pert:
            score = -lof.score_samples(
                input_pert.reshape(len(input_pert), -1).cpu().numpy()
            )
            score = th.from_numpy(score).float()
            score = 1 / score.clip(min=1)
            score = score.reshape(len(input_pert), -1).mean(-1)
            score_tpl += (score,)

        # Stack score_tpl and average
        score = th.stack(score_tpl).mean(0).unsqueeze(-1)

        return score

    return weights_fn
