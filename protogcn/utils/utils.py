import logging
import os.path as osp
import pickle
import numpy as np
import torch
import torch.optim as optim



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


def remap_model_keys(state_dict):
    """Remap old checkpoint key names to current model key names."""
    mapping = [
        ('gcn.pre.',   'mte.h_last.'),
        ('gcn.conv1.', 'mte.h_q.'),
        ('gcn.conv2.', 'mte.h_k.'),
        ('gcn.post.',  'mte.h_l.'),
        ('gcn.down.',  'mte.residual.'),
        ('gcn.bn.',    'mte.bn.'),
        ('gcn.A',      'mte.A'),
        ('gcn.alpha',  'mte.alpha'),
        ('gcn.beta',   'mte.beta'),
        ('tcn.add_coeff', 'tcn.add_coef'),
        ('prn.query_matrix.', 'prn.query.'),
        ('prn.memory_matrix.', 'prn.memory.'),
    ]
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in mapping:
            new_k = new_k.replace(old, new)
        new_sd[new_k]= v
    return new_sd


def build_scheduler(optimizer, cfg, total_iters, total_epochs):
    sched_cfg = cfg.get('scheduler', {})
    sched_type = sched_cfg.get('type', 'cosine_annealing')
    min_lr = sched_cfg.get('min_lr', 0.0)

    if sched_type == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters, eta_min=min_lr, last_epoch=-1
        )
    elif sched_type == 'cosine_warm_restarts':
        T_0 = sched_cfg.get('T_0', total_epochs // 3)
        T_mult = sched_cfg.get('T_mult', 1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
        )
    else:
        raise ValueError(f'Unknown scheduler type: {sched_type}')
    
    return scheduler, sched_type


def save_checkpoint(model, optimizer, scheduler, epoch, work_dir, filename, **kwargs):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        **kwargs
    }, osp.join(work_dir, filename))