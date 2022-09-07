import torch

from torch.nn import CosineSimilarity


def lime_weights(
    distance_mode: str = "cosine",
    kernel_width: float = 1.0,
):
    """
    Compute lime similarity weights given original and perturbed inputs.

    Args:
        distance_mode (str, optional): Mode to compute distance.
            Either ``'cosine'`` or ``'euclidean'``.
            Default: ``'cosine'``
        kernel_width (float, optional): Kernel width.
            Default: 1.0

    Returns:
        Callable: A function to compute weights given original and
            perturbed inputs.
    """

    def default_exp_kernel(inputs, inputs_pert):
        score_tpl = tuple()
        for original_inp, perturbed_inp in zip(inputs, inputs_pert):
            if distance_mode == "cosine":
                cos_sim = CosineSimilarity(dim=1)
                distance = 1 - cos_sim(
                    original_inp.reshape(len(original_inp), -1),
                    perturbed_inp.reshape(len(perturbed_inp), -1),
                )
            elif distance_mode == "euclidean":
                distance = torch.norm(
                    (original_inp - perturbed_inp).reshape(
                        len(original_inp), -1
                    ),
                    dim=1,
                )
            else:
                raise ValueError(
                    "distance_mode must be either cosine or euclidean."
                )

            score = (-1 * (distance**2) / (2 * (kernel_width**2))).exp()
            score_tpl += (score,)

        # Stack score_tpl and average
        score = torch.stack(score_tpl).mean(0).unsqueeze(-1)

        return score

    return default_exp_kernel
