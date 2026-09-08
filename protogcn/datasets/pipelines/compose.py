from collections.abc import Mapping, Sequence
import importlib


_SUBMODULES = ['sampling', 'formatting', 'pose_related', 'augmentations']

def _lookup(name):
    for mod_name in _SUBMODULES:
        mod = importlib.import_module(f'.{mod_name}', package=__package__)
        if hasattr(mod, name):
            return getattr(mod, name)
    raise KeyError(f"Unknown transform type: '{name}'")

class Compose:
    """
    Compose a data pipeline with a sequence of transforms.

    Args:
        transforms(list[dict | callable]):
            Either config dict of transforms of transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            elif isinstance(transform, Mapping):
                cfg = dict(transform)
                t_type=cfg.pop('type')
                self.transforms.append(_lookup(t_type)(**cfg))
            else:
                raise TypeError(f"transform must be callable, "
                                f"but got {type(transform)}")
            
    def __call__(self, data):
        """
        Call function to apply transforms sequentially.
        
        Args: 
            data (dict): A result dict contains the data to transform
        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
    
    def set_epoch(self, epoch):
        for t in self.transforms:
            if hasattr(t, 'set_epoch'):
                t.set_epoch(epoch)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string