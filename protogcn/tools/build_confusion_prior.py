import os 
import os.path as osp

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import build_dataloader, PoseDataset
from ..models import Recognizer, ProtoGCN, Head
from ..utils import get_logger, remap_model_keys


@hydra.main(config_path="../../configs", config_name="build_prior", version_base=None)
def main(cfg: DictConfig):
    ckpt_path = cfg.get('ckpt_path', None)
    assert ckpt_path is not None, "ckpt_path required: python -m protogcn.tools.build_confusion_prior ckpt_path=..."

    split = cfg.get('prior_split', 'val')
    out_path = cfg.get('out_path', None) or osp.join(
        osp.dirname(ckpt_path) or '.', f'confusion_prior_{split}.pt'
    )
    n_class = cfg['model']['cls_head']['num_classes']
    smooth_eps = float(cfg.get('smooth_eps', 1.0 / n_class))

    logger = get_logger('build_prior')
    logger.info(f'ckpt:         {ckpt_path}')
    logger.info(f'split:         {split}')
    logger.info(f'out:         {out_path}')
    logger.info(f'n_class:         {n_class}, smooth_eps: {smooth_eps:.4f}')


    # ---- Dataloader------
    data_cfg = cfg['data']
    assert split in data_cfg, f"split '{split}' not found in data config"
    dataset = PoseDataset(
        ann_file=data_cfg[split]['ann_file'],
        pipeline=data_cfg[split]['pipeline'],
        split=data_cfg[split].get('split'),
        data_prefix=data_cfg.get('data_prefix', ''),
        test_mode=True,
    )
    loader = build_dataloader(
        dataset,
        batch_size=data_cfg.get('test_dataloader', {}).get('video_per_gpu', 1),
        num_workers=data_cfg.get('workers_per_gpu', 4),
        shuffle=False,
    )
    logger.info(f'{split} samples: {len(dataset)}')

    # ─── Model (force prior OFF so baseline ckpt loads cleanly) ───
    model_cfg = cfg['model']
    backbone_cfg = {k: v for k, v in model_cfg['backbone'].items() if k != 'type'}
    head_cfg = {k: v for k, v in model_cfg['cls_head'].items() if k != 'type'}
    head_cfg['csc_prior_mode'] = 'off'
    head_cfg['csc_prior_path'] = None

    model = Recognizer(
        backbone=ProtoGCN(**backbone_cfg),
        cls_head=Head(**head_cfg),
        train_cfg=model_cfg.get('train_cfg'),
        test_cfg=model_cfg.get('test_cfg'),
    ).cuda()


    # ─── Load baseline ckpt ─────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = remap_model_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.info(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]}')
    if missing:
        # log_prior is expected to be missing — baseline has no prior buffer
        logger.info(f'Missing keys ({len(missing)}): {missing[:5]}')


    # ─── Inference ──────────────────────────────────────────
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc=f'{split} inference'):
            keypoint = data['keypoint'].cuda()
            label = data['label']
            if label.dim() > 1:
                label = label.squeeze(-1)
            label = label.numpy().reshape(-1)

            scores = model(keypoint, return_loss=False)
            scores = np.asarray(scores)        # (B, n_class)
            preds = scores.argmax(axis=-1)
            all_pred.extend(preds.tolist())
            all_true.extend(label.tolist())

    all_pred = np.asarray(all_pred, dtype=np.int64)
    all_true = np.asarray(all_true, dtype=np.int64)
    acc = float((all_pred == all_true).mean())
    logger.info(f'Baseline {split} acc: {acc:.4f}')


    # ----Confusion matrix & prior -----
    # C[i, j] = #(true=i, pred=j)
    C = np.zeros((n_class, n_class), dtype=np.float64)
    for t, p in zip(all_true, all_pred):
        C[t, p] += 1


    # Laplace smooth + row-normalize -> P(pred | true)
    P = (C + smooth_eps) / (C.sum(axis=1, keepdims=True) + n_class * smooth_eps)
    assert np.allclose(P.sum(axis=1), 1.0), 'row sums must equal 1'

    diag = np.diag(P)
    counts = np.bincount(all_true, minlength=n_class)
    logger.info(f'P diagonal: min={diag.min():.3f}, mean={diag.mean():.3f}, max={diag.max():.3f}')
    logger.info(f'weakest class: idx={int(diag.argmin())}, '
                f'P[y,y]={diag.min():.3f}, n={int(counts[diag.argmin()])}')
    logger.info(f'class count range: [{int(counts.min())}, {int(counts.max())}]')


    # ─── Save ───────────────────────────────────────────────
    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    torch.save({
        'prior':         torch.from_numpy(P).float(),    # (C, C), row-normalized
        'confusion_raw': torch.from_numpy(C).long(),     # pre-smoothing counts
        'class_counts':  torch.from_numpy(counts).long(),
        'n_class':       n_class,
        'split':         split,
        'smooth_eps':    smooth_eps,
        'baseline_ckpt': str(ckpt_path),
        'baseline_acc':  acc,
    }, out_path)
    logger.info(f'Saved → {out_path}')


if __name__ == '__main__':
    main()

