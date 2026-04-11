import os.path as osp

import numpy as np
import torch
from tqdm import tqdm


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
