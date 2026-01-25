import numpy as np
import torch
from collections.abc import Sequence


def to_tensor(data):
    """
    Convert object of various python types to :obj:`torch.Tensor`.

    Supported types are: 
            :class:`numpy.ndarray`, :class:`torch.Tensor`,
            :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class ToTensor:
    """
    Convert same values in results dict to `torch.Tensor` type in dataloader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """
        Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified 
                and passed to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
    

class Rename:
    """
    Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. 
            The key of the dict is the original name, while the value is the new name.
            If the original name not found in results, do nothing.
            Default: dict()
    """
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in results')
                results[value] = results[key]
                results.pop(key)

        return results
    

class Collect:
    """
    Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in ``meta_keys`` into
    a meta item called ``meta_name``. 
    This is usually the last stage of the data loader pipeline.
    
    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on ``meta_keys``.
            Default: ().
        nested (bool): If set as True, will apply data[x] = [data[x]] to all items in data.
            The arg is added for compatibility. Default: False.

    """
    def __init__(self,
                 keys,
                 meta_keys=(),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested
    
    def __call__(self, results):
        """
        Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed 
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]
        
        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = meta
        if self.nested:
            for k in data:
                data[k] = [data[k]]
        
        return data
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')
    
