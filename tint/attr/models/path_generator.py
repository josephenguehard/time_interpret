import numpy as np
import torch

from collections import defaultdict as ddict
from copy import deepcopy


def monotonic(vec1, vec2, vec3, ret="bool"):
    # check if vec3 [interpolation] is monotonic w.r.t. vec1 [baseline]
    # and vec2 [input]
    # i.e., vec3 should lie between vec1 and vec2 (for both +ve and -ve cases)

    increasing_dims = vec1 > vec2  # dims where baseline > input
    decreasing_dims = vec1 < vec2  # dims where baseline < input
    equal_dims = vec1 == vec2  # dims where baseline == input

    vec3_greater_vec1 = vec3 >= vec1
    vec3_greater_vec2 = vec3 >= vec2
    vec3_lesser_vec1 = vec3 <= vec1
    vec3_lesser_vec2 = vec3 <= vec2
    vec3_equal_vec1 = vec3 == vec1
    vec3_equal_vec2 = vec3 == vec2

    # if, for some dim: vec1 > vec2 then vec1 >= vec3 >= vec2
    # elif: vec1 < vec2 then vec1 <= vec3 <= vec2
    # elif: vec1 == vec2 then vec1 == vec3 == vec2
    valid = (
        increasing_dims * vec3_lesser_vec1 * vec3_greater_vec2
        + decreasing_dims * vec3_greater_vec1 * vec3_lesser_vec2
        + equal_dims * vec3_equal_vec1 * vec3_equal_vec2
    )

    if ret == "bool":
        return valid.sum() == vec1.shape[0]
    elif ret == "count":
        return valid.sum()
    elif ret == "vec":
        return valid


def make_monotonic_vec(vec1, vec2, vec3, steps):
    # create a new vec4 from vec3 [anchor] which is monotonic w.r.t. vec1
    # [baseline] and vec2 [input]

    mono_dims = monotonic(vec1, vec2, vec3, ret="vec")
    non_mono_dims = ~mono_dims

    if non_mono_dims.sum() == 0:
        return vec3

    # make vec4 monotonic
    vec4 = deepcopy(vec3)
    vec4[non_mono_dims] = vec2[non_mono_dims] - (1.0 / steps) * (
        vec2[non_mono_dims] - vec1[non_mono_dims]
    )

    return vec4


def distance(A, B):
    # return euclidean distance between two points
    return np.sqrt(np.sum((A - B) ** 2))


def find_next_wrd(
    wrd_idx,
    ref_idx,
    word_path,
    word_features,
    adj,
    strategy="greedy",
    steps=30,
):
    if wrd_idx == ref_idx:
        # If (for some reason) we do select the ref_idx as the previous
        # anchor word, then all further anchor words should be ref_idx
        return ref_idx

    anchor_map = ddict(list)
    cx = adj[wrd_idx].tocoo()

    for j, v in zip(cx.col, cx.data):
        # we should not consider the anchor word to be the ref_idx
        # [baseline] unless forced to.
        if j == ref_idx:
            continue

        if strategy == "greedy":
            # calculate the distance of the monotonized vec from the
            # anchor point
            monotonic_vec = make_monotonic_vec(
                word_features[ref_idx],
                word_features[wrd_idx],
                word_features[j],
                steps,
            )
            anchor_map[j] = [distance(word_features[j], monotonic_vec)]
        elif strategy == "maxcount":
            # count the number of non-monotonic dimensions (10000 is
            # an arbitrarily high and is a hack to be agnostic of
            # word_features dimension)
            non_mono_count = 10000 - monotonic(
                word_features[ref_idx],
                word_features[wrd_idx],
                word_features[j],
                ret="count",
            )
            anchor_map[j] = [non_mono_count]
        elif strategy == "non_monotonic":
            # Here we just use the distance between the reference
            # and the proposed word
            anchor_map[j] = [
                distance(word_features[ref_idx], word_features[j])
            ]
        else:
            raise NotImplementedError

    if len(anchor_map) == 0:
        return ref_idx

    sorted_dist_map = {
        k: v
        for k, v in sorted(anchor_map.items(), key=lambda item: item[1][0])
    }

    # remove words that are already selected in the path
    for key in word_path:
        sorted_dist_map.pop(key, None)

    if len(sorted_dist_map) == 0:
        return ref_idx

    # return the top key
    return next(iter(sorted_dist_map))


def find_word_path(
    wrd_idx,
    ref_idx,
    word_idx_map,
    word_features,
    adj,
    steps=30,
    strategy="greedy",
):
    # if wrd_idx is CLS or SEP then just copy that and return
    if ("[CLS]" in word_idx_map and wrd_idx == word_idx_map["[CLS]"]) or (
        "[SEP]" in word_idx_map and wrd_idx == word_idx_map["[SEP]"]
    ):
        return [wrd_idx] * (steps + 1)

    word_path = [wrd_idx]
    last_idx = wrd_idx
    for step in range(steps):
        next_idx = find_next_wrd(
            last_idx,
            ref_idx,
            word_path,
            word_features=word_features,
            adj=adj,
            strategy=strategy,
            steps=steps,
        )
        word_path.append(next_idx)
        last_idx = next_idx
    return word_path


def upscale(embs):
    # add a average embedding between each consecutive vec in embs
    embs = np.array(embs)
    avg_embs = 0.5 * (embs[0:-1] + embs[1:])
    final_embs = np.empty(
        (embs.shape[0] + avg_embs.shape[0], embs.shape[1]), dtype=embs.dtype
    )
    final_embs[::2] = embs
    final_embs[1::2] = avg_embs

    return final_embs


def make_monotonic_path(
    word_path_ids,
    ref_idx,
    word_features,
    steps=30,
    factor=0,
):
    monotonic_embs = [word_features[word_path_ids[0]]]
    vec1 = word_features[ref_idx]

    for idx in range(len(word_path_ids) - 1):
        vec2 = monotonic_embs[-1]
        vec3 = word_features[word_path_ids[idx + 1]]
        vec4 = make_monotonic_vec(vec1, vec2, vec3, steps)
        monotonic_embs.append(vec4)
    monotonic_embs.append(vec1)

    # reverse the list so that baseline is the first and input word is the last
    monotonic_embs.reverse()

    final_embs = monotonic_embs

    # do upscaling for factor number of times
    for _ in range(factor):
        final_embs = upscale(final_embs)

    # verify monotonicity
    check = True
    for i in range(len(final_embs) - 1):
        check *= monotonic(
            final_embs[-1], final_embs[i], final_embs[i + 1], ret="bool"
        )
    assert check

    return final_embs


def scale_inputs(
    input_ids,
    ref_input_ids,
    device,
    auxiliary_data,
    steps=30,
    factor=0,
    strategy="greedy",
):
    """
    Creates a monotonic path between input_ids and ref_input_ids
    (the baseline). This path is only composed of data points, which have been
    'monotonized'. The strategy used to build the path is either ``'greedy'``
    or ``'maxcount'``.

    Args:
        input_ids: The inputs.
        ref_input_ids: The baseline.
        device: Which device to use for the path.
        auxiliary_data: The knns previously computed.
        steps: Number of steps for the path. Default to 30
        factor: Up-scaling of the embeddings. Default to 0
        strategy: Strategy to build the path. Either ``'greedy'`` or
        ``'maxcount'``. Default to ``'greedy'``

    Returns:
        torch.Tensor: The monotonic path.

    References:
        #. `Discretized Integrated Gradients for Explaining Language Models <https://arxiv.org/abs/2108.13654>`_
        #. https://github.com/INK-USC/DIG
    """
    # generates the paths required by DIG
    word_idx_map, word_features, adj = auxiliary_data

    all_path_embs = []
    for idx in range(len(input_ids)):
        word_path = find_word_path(
            input_ids[idx],
            ref_input_ids[idx],
            word_idx_map=word_idx_map,
            word_features=word_features,
            adj=adj,
            steps=steps,
            strategy=strategy,
        )
        if strategy != "non_monotonic":
            embs = make_monotonic_path(
                word_path,
                ref_input_ids[idx],
                word_features=word_features,
                steps=steps,
                factor=factor,
            )
        else:
            embs = [word_features[idx] for idx in word_path]
            embs += [word_features[ref_input_ids[idx]]]
            embs.reverse()  # baseline --> input
        all_path_embs.append(embs)
    all_path_embs = torch.tensor(
        np.stack(all_path_embs, axis=1),
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )

    return all_path_embs
