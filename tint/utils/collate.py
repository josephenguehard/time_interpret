import collections
import re
import torch


np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch):
    r"""
    This method modifies the original collate function by only indexing tensors.
    Every other type of input (int, float, None) will be returned as it is.

    Args:
        batch: a single batch to be collated
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype)
                )

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    # If elem is a float, int, string or None,
    # the elem is returned without transformation
    elif isinstance(elem, (float, int, str)):
        return elem
    elif elem is None:
        return None
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: default_collate([d[key] for d in batch]) for key in elem}
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {
                key: default_collate([d[key] for d in batch]) for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(default_collate(samples) for samples in zip(*batch))
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                "each element in list of batch should be of equal size"
            )
        transposed = list(
            zip(*batch)
        )  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                default_collate(samples) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type(
                    [default_collate(samples) for samples in transposed]
                )
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
