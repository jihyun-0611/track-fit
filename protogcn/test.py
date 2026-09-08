import argparse
import logging
import os
import os.path as osp
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'protogcn'

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
import torch
from tqdm import tqdm

from .datasets import PoseDataset, build_dataloader
from .models import Head, ProtoGCN, Recognizer
from .utils import dump_file, get_logger, load_file, remap_model_keys

load_dotenv()


SCORE_SPLIT_ALIASES = {
    'val': ('val', 'val'),
    'internal': ('internal_test', 'internal'),
    'internal_test': ('internal_test', 'internal'),
    'final': ('final_test', 'final'),
    'final_test': ('final_test', 'final'),
}


@torch.no_grad()
def run_test(model, test_loader, test_dataset, eval_cfg, work_dir, logger, tag='test', label_map_file=None):
    """Run test inference and evaluate with extended metrics via dataset.evaluate().

    Collects softmax scores and pooled features in a single forward pass,
    then delegates all metric computation to dataset.evaluate().

    Args:
        model: Trained recognizer.
        test_loader: DataLoader for the test set.
        test_dataset: PoseDataset instance (test mode).
        eval_cfg: Evaluation config dict.
        work_dir: Directory to save confusion matrix PNG.
        logger: Logger instance.
        tag (str): Tag used in the confusion matrix filename ('last' or 'best').

    Returns:
        eval_results (dict): All computed metrics.
        all_scores (list[np.ndarray]): Per-sample softmax score vectors.
        cm_path (str): Path to saved confusion matrix PNG.
    """
    model.eval()
    all_scores, all_features, bs_nc_list = [], [], []

    raw_model = getattr(model, '_orig_mod', model)

    def _hook(module, inp, out):
        all_features.append(inp[0].detach().cpu().numpy())

    hook = raw_model.cls_head.fc_cls.register_forward_hook(_hook)
    try:
        for data in tqdm(test_loader, desc=f'Test ({tag})'):
            bs_nc_list.append((data['keypoint'].shape[0], data['keypoint'].shape[1]))
            preds = model(data['keypoint'].cuda(), return_loss=False)
            all_scores.extend(preds)
    finally:
        hook.remove()

    # Aggregate clip-level features → one vector per sample: (bs*nc, C) → (bs, C)
    features = np.concatenate(
        [f.reshape(bs, nc, -1).mean(axis=1) for (bs, nc), f in zip(bs_nc_list, all_features)],
        axis=0
    )

    cm_path = osp.join(work_dir, f'confusion_matrix_{tag}.png')
    metrics = list(eval_cfg.get('metrics', ['top_k_accuracy', 'mean_class_accuracy']))
    metrics += ['confusion_matrix', 'ece', 'intra_class_similarity']

    eval_results = test_dataset.evaluate(
        all_scores,
        metrics=metrics,
        metric_options={
            'top_k_accuracy': {'topk': tuple(eval_cfg.get('topk', (1, 5)))},
            'confusion_matrix': {'save_path': cm_path, 'label_map_file': label_map_file},
        },
        features=features,
        logger=logger,
    )

    return eval_results, all_scores, cm_path


def build_model_from_cfg(cfg, checkpoint, device='cuda:0', logger=None):
    """Build a recognizer from Hydra config and load a checkpoint."""
    model_cfg = cfg['model']
    backbone_cfg = {k: v for k, v in model_cfg['backbone'].items() if k != 'type'}
    head_cfg = {k: v for k, v in model_cfg['cls_head'].items() if k != 'type'}

    model = Recognizer(
        backbone=ProtoGCN(**backbone_cfg),
        cls_head=Head(**head_cfg),
        train_cfg=model_cfg.get('train_cfg'),
        test_cfg=model_cfg.get('test_cfg'),
    )

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    allowed_missing = ('cls_head.W', 'cls_head._current_epoch')
    allowed_unexpected = ('cls_head.W', 'cls_head._current_epoch')
    critical_missing = [key for key in missing if not key.startswith(allowed_missing)]
    critical_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected)]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            'Checkpoint does not match model config. '
            f'critical_missing={critical_missing[:10]}, '
            f'critical_unexpected={critical_unexpected[:10]}'
        )

    if logger is not None:
        if missing:
            logger.info('Ignored missing checkpoint keys (%d): %s', len(missing), missing[:5])
        if unexpected:
            logger.info('Ignored unexpected checkpoint keys (%d): %s', len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()
    return model


def normalize_state_dict_keys(state_dict):
    """Normalize common checkpoint prefixes and legacy ProtoGCN key names."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('_orig_mod.'):
            key = key[len('_orig_mod.'):]
        normalized[key] = value
    return remap_model_keys(normalized)


def resolve_dump_splits(split_names):
    """Map CLI split aliases to dataset split names and output file split names."""
    resolved = []
    for split_name in split_names:
        if split_name not in SCORE_SPLIT_ALIASES:
            valid = ', '.join(SCORE_SPLIT_ALIASES)
            raise ValueError(f"Unknown split '{split_name}'. Expected one of: {valid}")
        dataset_split, output_split = SCORE_SPLIT_ALIASES[split_name]
        if (dataset_split, output_split) not in resolved:
            resolved.append((dataset_split, output_split))
    return resolved


def build_dataset_for_split(cfg, dataset_split):
    """Build a test-mode dataset and force dataset-pkl split order."""
    data_cfg = cfg['data']
    cfg_key = 'val' if dataset_split == 'val' else 'test'
    if cfg_key not in data_cfg:
        raise KeyError(f"data.{cfg_key} config is required for split '{dataset_split}'")

    split_cfg = data_cfg[cfg_key]
    ann_file = split_cfg['ann_file']
    dataset = PoseDataset(
        ann_file=ann_file,
        pipeline=split_cfg['pipeline'],
        split=dataset_split,
        data_prefix=data_cfg.get('data_prefix', ''),
        test_mode=True,
    )
    reorder_dataset_to_split_order(
        dataset,
        ann_file=ann_file,
        dataset_split=dataset_split,
        data_prefix=data_cfg.get('data_prefix', ''),
    )
    return dataset


def reorder_dataset_to_split_order(dataset, ann_file, dataset_split, data_prefix=''):
    """Reorder PoseDataset.video_infos to match dataset pkl split list exactly."""
    payload = load_file(ann_file)
    if not isinstance(payload, dict) or 'split' not in payload or 'annotations' not in payload:
        raise ValueError(f"Annotation file must contain split and annotations: {ann_file}")
    if dataset_split not in payload['split']:
        raise KeyError(f"Split '{dataset_split}' not found in annotation file: {ann_file}")

    split_frame_dirs = list(payload['split'][dataset_split])
    if not dataset.video_infos:
        raise ValueError(f"Dataset split '{dataset_split}' has no samples")

    identifier = 'filename' if 'filename' in dataset.video_infos[0] else 'frame_dir'
    expected_ids = [osp.join(data_prefix, frame_dir) for frame_dir in split_frame_dirs]
    by_id = {}
    for item in dataset.video_infos:
        sample_id = item[identifier]
        if sample_id in by_id:
            raise AssertionError(f"Duplicate sample id after dataset build: {sample_id}")
        by_id[sample_id] = item

    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    if missing:
        raise AssertionError(
            f"Dataset split order cannot be matched for '{dataset_split}'. "
            f"Missing {len(missing)} samples; first missing: {missing[0]}"
        )

    dataset.video_infos = [by_id[sample_id] for sample_id in expected_ids]
    actual_ids = [item[identifier] for item in dataset.video_infos]
    assert actual_ids == expected_ids, (
        f"Dataset order mismatch for split '{dataset_split}' after reordering"
    )


@torch.no_grad()
def dump_scores_for_split(model, loader, device, score_kind='prob'):
    """Return one (N, num_classes) array from a score dump dataloader."""
    if score_kind not in ('prob', 'logit'):
        raise ValueError("score_kind must be 'prob' or 'logit'")

    raw_model = getattr(model, '_orig_mod', model)
    raw_model.test_cfg['average_clips'] = 'prob' if score_kind == 'prob' else 'score'

    chunks = []
    for data in tqdm(loader, desc='Dump scores'):
        scores = model(data['keypoint'].to(device), return_loss=False)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 2:
            raise AssertionError(f'Model output must have shape (B, C); got {scores.shape}')
        chunks.append(scores)

    if not chunks:
        raise ValueError('No score batches were produced')
    scores = np.concatenate(chunks, axis=0)
    if score_kind == 'prob':
        row_sums = scores.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), (
            f'Softmax score row sums must be 1.0; '
            f'range=[{row_sums.min():.6f}, {row_sums.max():.6f}]'
        )
    return scores


def dump_scores_main(cfg, args, overrides):
    logger = get_logger('dump_scores')
    logger.setLevel(logging.INFO)

    checkpoint = args.checkpoint or cfg.get('load_from')
    if not checkpoint:
        raise ValueError('Checkpoint required: pass --checkpoint or set load_from=...')
    checkpoint = osp.abspath(osp.expanduser(str(checkpoint)))
    if not osp.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    score_model = args.score_model or Path(checkpoint).parent.name
    if not score_model:
        raise ValueError('Could not infer score model name; pass --score-model')

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'{device} requested but CUDA is not available')

    logger.info('Config overrides: %s', overrides)
    logger.info('Checkpoint: %s', checkpoint)
    logger.info('Score model: %s', score_model)
    logger.info('Score kind: %s', args.score_kind)

    model = build_model_from_cfg(cfg, checkpoint, device=device, logger=logger)
    output_dir = Path(args.score_dir) / score_model
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg['data']
    num_workers = int(data_cfg.get('workers_per_gpu', 4))
    batch_size = int(data_cfg.get('test_dataloader', {}).get('video_per_gpu', 1))

    written = {}
    for dataset_split, output_split in resolve_dump_splits(args.splits):
        dataset = build_dataset_for_split(cfg, dataset_split)
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=device.startswith('cuda'),
            persistent_workers=num_workers > 0,
        )
        scores = dump_scores_for_split(model, loader, device=device, score_kind=args.score_kind)
        assert scores.shape[0] == len(dataset), (
            f"Score row count for split '{dataset_split}' must match dataset length: "
            f'{scores.shape[0]} != {len(dataset)}'
        )
        out_path = output_dir / f'{output_split}.pkl'
        dump_file(scores, str(out_path))
        written[output_split] = str(out_path)
        logger.info(
            'Wrote %s | split=%s rows=%d classes=%d',
            out_path, dataset_split, scores.shape[0], scores.shape[1],
        )

    return written


def load_hydra_config(config_name, overrides):
    config_dir = Path(__file__).resolve().parents[1] / 'configs'
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate ProtoGCN checkpoints and dump per-split score pkl files.'
    )
    parser.add_argument(
        '--dump-scores',
        action='store_true',
        help='Dump val/internal/final score pkl files for run_ensemble.py.',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint path. Defaults to resolved cfg.load_from.',
    )
    parser.add_argument(
        '--score-dir',
        default='scores',
        help='Output root. Files are written to {score_dir}/{score_model}/{split}.pkl.',
    )
    parser.add_argument(
        '--score-model',
        '--model-name',
        dest='score_model',
        default=None,
        help='Model directory name under --score-dir, e.g. protect, b_t2, jm_t2w.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['val', 'internal', 'final'],
        help='Splits to dump. Aliases: val, internal/internal_test, final/final_test.',
    )
    parser.add_argument(
        '--score-kind',
        choices=['prob', 'logit'],
        default='prob',
        help='Dump softmax probabilities or averaged logits. Use prob for run_ensemble.py.',
    )
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Inference device, e.g. cuda:0 or cpu.',
    )
    parser.add_argument(
        '--config-name',
        default='config',
        help='Hydra config name under configs/ without .yaml.',
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args, overrides = parser.parse_known_args(argv)
    if not args.dump_scores:
        parser.error('Only --dump-scores mode is currently implemented.')

    try:
        cfg = load_hydra_config(args.config_name, overrides)
        written = dump_scores_main(cfg, args, overrides)
    except (
        AssertionError,
        FileNotFoundError,
        HydraException,
        KeyError,
        OmegaConfBaseException,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f'ERROR: {exc}', flush=True)
        return 1

    print('Score dumps written:')
    for split, path in written.items():
        print(f'  {split}: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
