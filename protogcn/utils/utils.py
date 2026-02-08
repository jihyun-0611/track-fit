import logging
import os.path as osp
import pickle
import numpy as np


def get_logger(name='protogcn', log_level=logging.INFO):
    """Get logger by name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
    return logger


def load_file(filepath):
    """Load data from pickle or json file"""
    if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f'Unsupported file format: {filepath}')
    

def dump_file(data, filepath):
    """Dump data to pickle or json file"""
    if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f)
    else:
        raise ValueError(f'Unsupported file format: {filepath}')


def auto_mix2(results_list, weights=None):
    """Mix two modality  results with various weights.
    
    Args:
        results_list: [rgb_results, pose_results], each is list of (num_classes,) arrays
        weights: If None, try weights from 0.0 to 1.0 with step 0.05

    Returns:
        dict: {weight_str: mixed_results}
    """
    if weights is None:
        weights = [i / 20.0 for i in range(21)] # 0.0, 0.05, ..., 1.0

    rgb = np.array(results_list[0])
    pose = np.array(results_list[1])

    preds = {}
    for w in weights:
        mixed = w * rgb + (1-w) *pose
        key = f'{w:.2f}'
        preds[key] = list(mixed)

    return preds

