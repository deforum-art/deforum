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


def parse_seed_for_mode(current_seed, mode, seed_list=None):
    if mode == "ladder":
        assert seed_list is not None, "seed_list cannot be None for mode 'ladder'"
    if mode == "random":
        return np.random.randint(0, (2**16) - 1)
    elif mode == "iter":
        current_seed = current_seed + 1
        return current_seed
    elif mode == "constant":
        return current_seed
    elif mode == "ladder":
        first = seed_list.pop(0)
        seed_list.append(first)
        return seed_list[0]
    else:
        raise ValueError(f"Invalid seed mode {mode}")
