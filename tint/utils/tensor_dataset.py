from torch import Tensor
from torch.utils.data import Dataset, default_collate
from typing import Tuple


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *inputs) -> None:
        tensors = tuple(x for x in inputs if isinstance(x, Tensor))
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.inputs = inputs

    def __getitem__(self, index):
        return tuple(x[index] if isinstance(x, Tensor) else x for x in self.inputs)

    def __len__(self):
        return self.tensors[0].size(0)
