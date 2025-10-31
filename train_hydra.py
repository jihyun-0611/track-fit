#!/usr/bin/env python
"""
Hydra-powered training script for Track-Fit ProtoGCN fine-tuning

Usage:
    # Default training
    python train_hydra.py

    # Phase 1 (freeze backbone)
    python train_hydra.py experiment=phase1_freeze

    # Phase 2 (full fine-tuning)
    python train_hydra.py experiment=phase2_finetune

    # Override parameters
    python train_hydra.py experiment=phase1_freeze training.epochs=30 training.optimizer.lr=0.02

    # Multiple runs (hyperparameter search)
    python train_hydra.py -m training.optimizer.lr=0.001,0.01,0.1

    # Custom config and GPU
    python train_hydra.py mmcv_config=configs/exercise/j.py training.gpus=2
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from mmcv import Config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
PROTOGCN_DIR = PROJECT_ROOT / 'external' / 'ProtoGCN'

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROTOGCN_DIR))


def update_mmcv_config(cfg: Config, hydra_cfg: DictConfig) -> Config:
    """Update mmcv config with Hydra overrides"""

    # Update work_dir
    if 'experiment' in hydra_cfg and 'work_dir' in hydra_cfg.experiment:
        cfg.work_dir = str(hydra_cfg.experiment.work_dir)

    # Update total_epochs
    if 'training' in hydra_cfg and 'epochs' in hydra_cfg.training:
        cfg.total_epochs = hydra_cfg.training.epochs

    # Update optimizer
    if 'training' in hydra_cfg and 'optimizer' in hydra_cfg.training:
        opt_cfg = OmegaConf.to_container(hydra_cfg.training.optimizer, resolve=True)
        if opt_cfg:
            cfg.optimizer.update(opt_cfg)

    # Update learning rate config
    if 'training' in hydra_cfg and 'lr_config' in hydra_cfg.training:
        lr_cfg = OmegaConf.to_container(hydra_cfg.training.lr_config, resolve=True)
        if lr_cfg and hasattr(cfg, 'lr_config'):
            cfg.lr_config.update(lr_cfg)

    # Update batch size
    if 'training' in hydra_cfg and 'batch_size' in hydra_cfg.training:
        cfg.data.videos_per_gpu = hydra_cfg.training.batch_size

    # Update workers
    if 'training' in hydra_cfg and 'workers' in hydra_cfg.training:
        cfg.data.workers_per_gpu = hydra_cfg.training.workers

    # Update checkpoint interval
    if 'training' in hydra_cfg and 'checkpoint' in hydra_cfg.training:
        if 'interval' in hydra_cfg.training.checkpoint:
            cfg.checkpoint_config.interval = hydra_cfg.training.checkpoint.interval

    # Update evaluation interval
    if 'training' in hydra_cfg and 'evaluation' in hydra_cfg.training:
        if 'interval' in hydra_cfg.training.evaluation:
            cfg.evaluation.interval = hydra_cfg.training.evaluation.interval

    # Update log interval
    if 'training' in hydra_cfg and 'log_interval' in hydra_cfg.training:
        cfg.log_config.interval = hydra_cfg.training.log_interval

    # Update pretrained model path
    if 'pretrained' in hydra_cfg:
        pretrained_path = str(hydra_cfg.pretrained) if hydra_cfg.pretrained else None
        if pretrained_path and os.path.exists(pretrained_path):
            cfg.load_from = pretrained_path

    # Update model config if provided
    if 'model' in hydra_cfg:
        if 'num_prototype' in hydra_cfg.model:
            cfg.model.backbone.num_prototype = hydra_cfg.model.num_prototype
        if 'num_classes' in hydra_cfg.model:
            cfg.model.cls_head.num_classes = hydra_cfg.model.num_classes
        if 'dropout' in hydra_cfg.model:
            cfg.model.cls_head.dropout = hydra_cfg.model.dropout

    return cfg


def run_distributed_training(
    config_path: str,
    gpus: int,
    port: Optional[int],
    validate: bool,
    test_best: bool,
    test_last: bool,
    launcher: str = 'pytorch',
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> int:
    """Run distributed training using torch.distributed.launch"""

    # Generate random port if not specified
    if port is None:
        import random
        port = 12000 + random.randint(0, 20000)

    # Build command
    cmd = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={gpus}',
        f'--master_port={port}',
        str(PROTOGCN_DIR / 'tools' / 'train.py'),
        config_path,
        f'--launcher={launcher}',
    ]

    if validate:
        cmd.append('--validate')
    if test_best:
        cmd.append('--test-best')
    if test_last:
        cmd.append('--test-last')
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    if deterministic:
        cmd.append('--deterministic')

    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROTOGCN_DIR}:{env.get('PYTHONPATH', '')}"
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Can be made configurable

    # Run training
    print("=" * 60)
    print("ProtoGCN Fine-tuning with Hydra")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"GPUs: {gpus}")
    print(f"Port: {port}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()

    result = subprocess.run(cmd, env=env, cwd=str(PROTOGCN_DIR))

    print()
    print("=" * 60)
    print("Training completed" if result.returncode == 0 else "Training failed")
    print("=" * 60)

    return result.returncode


@hydra.main(version_base=None, config_path="configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra"""

    # Print configuration
    print("\n" + "=" * 60)
    print("Hydra Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    # Load mmcv config
    mmcv_config_path = Path(cfg.mmcv_config)
    if not mmcv_config_path.exists():
        raise FileNotFoundError(f"MMCv config not found: {mmcv_config_path}")

    print(f"Loading MMCv config from: {mmcv_config_path}")
    mmcv_cfg = Config.fromfile(str(mmcv_config_path))

    # Update mmcv config with Hydra overrides
    print("Applying Hydra overrides to MMCv config...")
    mmcv_cfg = update_mmcv_config(mmcv_cfg, cfg)

    # Save updated config to temporary file
    temp_config_path = Path(mmcv_cfg.work_dir) / 'hydra_mmcv_config.py'
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove problematic attributes before dumping
    # These are imported classes/objects from the config file that cause yapf parsing errors
    attrs_to_remove = ['Path', '_data_dir', '_work_dir', '_checkpoint_dir', 'load_dotenv']
    for attr in attrs_to_remove:
        # mmcv Config stores attributes in _cfg_dict
        if hasattr(mmcv_cfg, '_cfg_dict') and attr in mmcv_cfg._cfg_dict:
            mmcv_cfg._cfg_dict.pop(attr, None)
        elif hasattr(mmcv_cfg, attr):
            try:
                delattr(mmcv_cfg, attr)
            except (AttributeError, TypeError):
                pass

    mmcv_cfg.dump(str(temp_config_path))

    print(f"Saved updated config to: {temp_config_path}")
    print(f"Work directory: {mmcv_cfg.work_dir}")
    print()

    # Run distributed training
    exit_code = run_distributed_training(
        config_path=str(temp_config_path),
        gpus=cfg.training.gpus,
        port=cfg.training.port,
        validate=cfg.training.validate,
        test_best=cfg.training.test_best,
        test_last=cfg.training.test_last,
        launcher=cfg.training.launcher,
        seed=cfg.training.seed,
        deterministic=cfg.training.deterministic,
    )

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
