import math
import numpy as np
import platform
import random
import torch
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader


# Increase file descriptor limit on non-Windows systems
if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def worker_init_fn(worker_id, num_workers, rank, seed):
    """
    Init the random seed for various workers.

    The seed of each worker equals to:
    num_worker * rank + worker_id + user_seed
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    """Custom collate function for skeleton data.
    
    Handles nested dicts and converts numpy arrays to tensors.

    Args:
        batch(list): List of samples from dataset.
    
    Returns:
        dict: Collated batch data.
    """
    if not isinstance(batch, list):
        raise TypeError(f'batch must be a list, but got {type(batch)}')
    
    if len(batch) == 0:
        raise ValueError('batch is empty')
    
    # check if batch items are dicts.
    if isinstance(batch[0], dict):
        collated = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]

            # Stack tensors/arrays
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                collated[key] = torch.from_numpy(np.stack(values, axis=0))
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            elif isinstance(values[0], dict):
                # Nested dict - recursively collate
                collated[key] = collate_fn(values)
            else:
                # Keep as list for non-stackable types
                collated[key] = values
        return collated
    
    # Handle non-dict batch (e.g., tuples)
    elif isinstance(batch[0], (tuple, list)):
        transposed = list(zip(*batch))
        return [collate_fn(samples) for samples in transposed]
    
    # Handle tensor/array batch
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], np.ndarray):
        return torch.from_numpy(np.stack(batch, axis=0))
    elif isinstance(batch[0], (int, float)):
        return torch.tensor(batch)
    else:
        return batch
    

def build_dataloader(dataset, 
                     batch_size,
                     num_workers=4,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=False,
                     **kwargs):
    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        seed=seed
    ) if seed is not None else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=init_fn,
        **kwargs
    )

