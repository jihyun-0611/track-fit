import numpy as np


class RepeatDataset:
    """
    A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` larger than the original dataset.
    This is useful when the data loading time is long but the dataset is small.
    Using RepeatDataset can reduce the data loading time beween epochs.

    Args:
        dataset (Dataset): The dataset to be repeated.
        times (int): Repeat times.
    """
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        if hasattr(dataset, 'class_prob'):
            self.class_prob = dataset.class_prob
        
        self._ori_len = len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]
    
    def __len__(self):
        return self.times * self._ori_len
    
    @property
    def video_infos(self):
        """Access video_infos from underlying dataset."""
        return self.dataset.video_infos
    

class ConcatDataset:
    """
    A wrapper of concatenated dataset.

    The length of concatenated dataset will be the sum of length of all datasets.
    This is useful when you want to train a model with multiple data source.

    Args:
        datasets (list[Dataset]): List of datasets to be concatenated.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(x) for x in self.datasets]
        self.cumsum = np.cumsum(self.lens)
    
    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumsum, idx, side='right')
        if dataset_idx == 0:
            item_idx = idx
        else:
            item_idx = idx - self.cumsum[dataset_idx - 1]
        return self.datasets[dataset_idx][item_idx]
    
    def __len__(self):
        return sum(self.lens)
    
    @property
    def video_infos(self):
        """Access combined video_infos from all datasets."""
        infos =[]
        for ds in self.datasets:
            infos.extend(ds.video_infos)
        return infos
