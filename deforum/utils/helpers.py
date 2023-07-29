import numpy as np
import torch


def get_at_index_if_list_else_first(item, index=0):
    if isinstance(item, (list, tuple)):
        assert len(item) > index, f"Index {index} is out of bounds for list of length {len(item)}"
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            raise ValueError("Cannot get item from empty list")
        return item[index]
    elif isinstance(item, (torch.Tensor, np.ndarray)):
        if item.ndim == 4:
            # if (N, C, H, W)
            if item.shape[1] == 3:
                item = item.permute(0, 2, 3, 1)
            return item[index]
        else:
            return item
    else:
        return item
