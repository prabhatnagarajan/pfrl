
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import copy
import numpy as np
import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map, np_str_obj_array_pattern


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    """Forked from: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L216
    """
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
    return collate([torch.tensor(b) for b in batch], collate_fn_map=collate_fn_map)

pfrl_default_collate_fn_map = copy.deepcopy(default_collate_fn_map)
pfrl_default_collate_fn_map[np.ndarray] = collate_numpy_array_fn

def _to_recursive(batched: Any, device: torch.device) -> Any:
    if isinstance(batched, torch.Tensor):
        return batched.to(device)
    elif isinstance(batched, list):
        return [x.to(device) for x in batched]
    elif isinstance(batched, tuple):
        return tuple(x.to(device) for x in batched)
    else:
        raise TypeError("Unsupported type of data")


def batch_states(
    states: Sequence[Any], device: torch.device, phi: Callable[[Any], Any]
) -> Any:
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        device (module): CPU or GPU the data should be placed on
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    features = [phi(s) for s in states]
    collated_features = collate(batch, collate_fn_map=pfrl_default_collate_fn_map)
    if isinstance(features[0], tuple):
        collated_features = tuple(collated_features)
    return _to_recursive(collated_features, device)
