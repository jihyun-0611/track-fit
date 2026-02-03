import logging
import os.path as osp
import pickle


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

