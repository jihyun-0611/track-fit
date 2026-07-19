#!/usr/bin/env python3
"""Post-hoc ensemble runner for saved softmax score pickles."""

from __future__ import annotations

import argparse
import itertools
import os
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


NUM_CLASSES = 22
EPS = 1e-12
SPLIT_TO_DATASET_SPLIT = {
    "val": "val",
    "internal": "internal_test",
    "final": "final_test",
}
EXPECTED_SPLIT_LENGTHS = {
    "val": 237,
    "internal": 236,
    "final": 61,
}
KNOWN_MODELS = {"pt05", "b_T2", "jm_T2w", "a_basic"}
WEIGHT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
ADD_A_WEIGHTS = (0.2, 0.3)


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _score_path(score_dir: str, model: str, split: str) -> Path:
    return Path(score_dir) / model / f"{split}.pkl"


def _validate_split(split: str) -> None:
    if split not in SPLIT_TO_DATASET_SPLIT:
        valid = ", ".join(SPLIT_TO_DATASET_SPLIT)
        raise ValueError(f"Unknown split '{split}'. Expected one of: {valid}")


def _coerce_score_array(raw_scores, *, source: Path, split: str) -> np.ndarray:
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score file {source} cannot be converted to a float array") from exc

    expected_n = EXPECTED_SPLIT_LENGTHS[split]
    assert scores.ndim == 2, (
        f"Score file {source} must have shape (N, {NUM_CLASSES}); "
        f"got {scores.shape}"
    )
    assert scores.shape == (expected_n, NUM_CLASSES), (
        f"Score file {source} has shape {scores.shape}; "
        f"expected {expected_n} rows and {NUM_CLASSES} classes for split '{split}'"
    )
    assert np.all(np.isfinite(scores)), f"Score file {source} contains NaN or inf"
    assert np.all(scores >= -1e-8), f"Score file {source} contains negative probabilities"
    row_sums = scores.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"Score file {source} must contain softmax probabilities; "
        f"row sums range from {row_sums.min():.6f} to {row_sums.max():.6f}"
    )
    return np.clip(scores, 0.0, 1.0)


def _extract_scores_and_frame_dirs(payload, *, source: Path, split: str):
    frame_dirs = None
    raw_scores = payload

    if isinstance(payload, dict):
        score_keys = ("scores", "probs", "probabilities", "score")
        for key in score_keys:
            if key in payload:
                raw_scores = payload[key]
                break
        else:
            raise ValueError(
                f"Score file {source} is a dict but has none of these score keys: "
                f"{', '.join(score_keys)}"
            )
        if "frame_dirs" in payload:
            frame_dirs = list(payload["frame_dirs"])

    return _coerce_score_array(raw_scores, source=source, split=split), frame_dirs


def _load_scores_with_frame_dirs(score_dir: str, model: str, split: str):
    _validate_split(split)
    if model not in KNOWN_MODELS:
        valid = ", ".join(sorted(KNOWN_MODELS))
        raise ValueError(f"Unknown model '{model}'. Expected one of: {valid}")

    path = _score_path(score_dir, model, split)
    try:
        payload = _load_pickle(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Score file not found for {model}/{split}: {path}") from exc
    return _extract_scores_and_frame_dirs(payload, source=path, split=split)


def load_scores(score_dir: str, model: str, split: str) -> np.ndarray:
    """Load one model/split score matrix with shape (N, 22)."""
    scores, _ = _load_scores_with_frame_dirs(score_dir, model, split)
    return scores


def load_labels(dataset_pkl: str, split: str) -> tuple[np.ndarray, list[str]]:
    """Return labels and frame_dirs in the dataset split order."""
    _validate_split(split)
    dataset_path = Path(dataset_pkl)
    dataset = _load_pickle(dataset_path)

    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset pickle {dataset_path} must contain a dict")
    if "split" not in dataset or "annotations" not in dataset:
        raise ValueError(f"Dataset pickle {dataset_path} must contain 'split' and 'annotations'")

    dataset_split = SPLIT_TO_DATASET_SPLIT[split]
    split_dict = dataset["split"]
    annotations = dataset["annotations"]
    if dataset_split not in split_dict:
        raise ValueError(f"Dataset split '{dataset_split}' is missing from {dataset_path}")
    if not isinstance(annotations, list):
        raise ValueError(f"Dataset annotations in {dataset_path} must be a list")

    frame_dirs = list(split_dict[dataset_split])
    expected_n = EXPECTED_SPLIT_LENGTHS[split]
    assert len(frame_dirs) == expected_n, (
        f"Dataset split '{dataset_split}' has {len(frame_dirs)} samples; "
        f"expected {expected_n} for split '{split}'"
    )

    annotation_by_frame_dir = {}
    for annotation in annotations:
        if "frame_dir" not in annotation or "label" not in annotation:
            raise ValueError(
                f"Every annotation in {dataset_path} must contain 'frame_dir' and 'label'"
            )
        frame_dir = annotation["frame_dir"]
        assert frame_dir not in annotation_by_frame_dir, (
            f"Duplicate annotation for frame_dir '{frame_dir}' in {dataset_path}"
        )
        annotation_by_frame_dir[frame_dir] = annotation

    missing = [frame_dir for frame_dir in frame_dirs if frame_dir not in annotation_by_frame_dir]
    assert not missing, (
        f"{len(missing)} frame_dir values from split '{dataset_split}' are missing "
        f"from annotations; first missing: {missing[0] if missing else 'none'}"
    )

    labels = np.asarray(
        [annotation_by_frame_dir[frame_dir]["label"] for frame_dir in frame_dirs],
        dtype=np.int64,
    )
    assert labels.ndim == 1 and labels.shape[0] == len(frame_dirs)
    assert np.all((0 <= labels) & (labels < NUM_CLASSES)), (
        f"Labels for split '{split}' must be in [0, {NUM_CLASSES - 1}]"
    )
    return labels, frame_dirs


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def calibrate(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities via log-probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs to be a 2D array, got shape {probs.shape}")
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got {T}")

    logits = np.log(np.clip(probs, EPS, None)) / float(T)
    return _softmax(logits)


def fit_temperature(probs_val: np.ndarray, labels_val: np.ndarray) -> float:
    """Fit temperature on val only by minimizing NLL over a fixed grid."""
    probs_val = np.asarray(probs_val, dtype=np.float64)
    labels_val = np.asarray(labels_val, dtype=np.int64)
    _assert_probs_and_labels(probs_val, labels_val)

    best_temperature = None
    best_nll = math.inf
    for temperature in np.round(np.arange(0.5, 5.0 + 1e-9, 0.1), 1):
        calibrated = calibrate(probs_val, float(temperature))
        nll = -np.log(calibrated[np.arange(labels_val.shape[0]), labels_val] + EPS).mean()
        if nll < best_nll:
            best_nll = float(nll)
            best_temperature = float(temperature)

    assert best_temperature is not None
    return best_temperature


def fuse(prob_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Fuse probability matrices by weighted averaging and row normalization."""
    if not prob_list:
        raise ValueError("prob_list must not be empty")
    if len(prob_list) != len(weights):
        raise ValueError(f"Got {len(prob_list)} probability arrays but {len(weights)} weights")

    arrays = [np.asarray(probs, dtype=np.float64) for probs in prob_list]
    shape = arrays[0].shape
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError("Every probability array must be 2D")
    if any(arr.shape != shape for arr in arrays):
        shapes = [arr.shape for arr in arrays]
        raise ValueError(f"All probability arrays must have the same shape; got {shapes}")

    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0):
        raise ValueError(f"Weights must be non-negative, got {weights}")
    weight_sum = weights_arr.sum()
    if weight_sum <= 0:
        raise ValueError("At least one weight must be positive")

    fused = np.zeros(shape, dtype=np.float64)
    for probs, weight in zip(arrays, weights_arr):
        fused += probs * weight
    fused /= weight_sum

    row_sums = fused.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Fused probabilities contain a row with zero total mass")
    return fused / row_sums


def evaluate(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate top-1, top-5, mean class accuracy, and 15-bin ECE."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _assert_probs_and_labels(probs, labels)

    pred = probs.argmax(axis=1)
    top1 = float((pred == labels).mean())

    top_k = min(5, probs.shape[1])
    topk = np.argpartition(-probs, kth=top_k - 1, axis=1)[:, :top_k]
    top5 = float((topk == labels[:, None]).any(axis=1).mean())

    recalls = []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        recalls.append(float((pred[cls_mask] == cls).mean()))
    mca = float(np.mean(recalls)) if recalls else 0.0

    ece = _expected_calibration_error(probs, labels, pred, n_bins=15)
    return {"top1": top1, "top5": top5, "mca": mca, "ece": ece}


def sample_diff(pred_a, pred_b, labels, frame_dirs) -> dict:
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    frame_dirs = list(frame_dirs)

    assert pred_a.shape == pred_b.shape == labels.shape, (
        f"Prediction and label shapes must match; got "
        f"{pred_a.shape}, {pred_b.shape}, {labels.shape}"
    )
    assert len(frame_dirs) == labels.shape[0], (
        f"frame_dirs length {len(frame_dirs)} does not match labels length {labels.shape[0]}"
    )

    a_correct = pred_a == labels
    b_correct = pred_b == labels
    return {
        "a_only_correct": [
            frame_dir
            for frame_dir, only_correct in zip(frame_dirs, a_correct & ~b_correct)
            if only_correct
        ],
        "b_only_correct": [
            frame_dir
            for frame_dir, only_correct in zip(frame_dirs, ~a_correct & b_correct)
            if only_correct
        ],
    }


def mcnemar_exact(pred_a, pred_b, labels) -> float:
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    assert pred_a.shape == pred_b.shape == labels.shape, (
        f"Prediction and label shapes must match; got "
        f"{pred_a.shape}, {pred_b.shape}, {labels.shape}"
    )

    a_correct = pred_a == labels
    b_correct = pred_b == labels
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    n = n01 + n10
    if n == 0:
        return 1.0

    tail = min(n01, n10)
    cdf = sum(math.comb(n, k) for k in range(tail + 1)) / float(2**n)
    return min(1.0, 2.0 * cdf)


def save_confusion_matrix_image(
    pred: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    *,
    label_map_file: str | None = None,
    title: str = "Confusion Matrix",
) -> str:
    """Save a row-normalized confusion matrix heatmap image."""
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    assert pred.shape == labels.shape, (
        f"Prediction and label shapes must match; got {pred.shape}, {labels.shape}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for true_label, pred_label in zip(labels, pred):
        cm[int(true_label), int(pred_label)] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        cm = cm / cm.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm)

    tick_labels = _confusion_matrix_tick_labels(label_map_file)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _assert_probs_and_labels(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs to be 2D, got shape {probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels to be 1D, got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of probability rows {probs.shape[0]} does not match "
            f"number of labels {labels.shape[0]}"
        )
    if probs.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty split")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError(
            f"Labels must be in [0, {probs.shape[1] - 1}] for probs shape {probs.shape}"
        )


def _confusion_matrix_tick_labels(label_map_file: str | None) -> list[str]:
    if not label_map_file:
        return [str(idx) for idx in range(NUM_CLASSES)]

    path = Path(label_map_file)
    if not path.exists():
        raise FileNotFoundError(f"Label map file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_name = {}
    for key, value in label_map.items():
        try:
            idx = int(value)
            name = str(key)
        except (TypeError, ValueError):
            idx = int(key)
            name = str(value)
        idx_to_name[idx] = name
    return [idx_to_name.get(idx, str(idx)) for idx in range(NUM_CLASSES)]


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    pred: np.ndarray,
    *,
    n_bins: int,
) -> float:
    confidences = probs.max(axis=1)
    correct = pred == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for bin_idx in range(n_bins):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidences[in_bin].mean())
        ece += float(in_bin.mean()) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _assert_score_alignment(
    *,
    model: str,
    split: str,
    scores: np.ndarray,
    labels: np.ndarray,
    frame_dirs: list[str],
    score_frame_dirs: list[str] | None,
) -> None:
    assert scores.shape[0] == labels.shape[0] == len(frame_dirs), (
        f"Cannot align scores for {model}/{split}: score rows={scores.shape[0]}, "
        f"labels={labels.shape[0]}, frame_dirs={len(frame_dirs)}"
    )
    if score_frame_dirs is not None:
        assert score_frame_dirs == frame_dirs, (
            f"Score frame_dir order for {model}/{split} does not match dataset split order"
        )


def _unique_models(models: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for model in models:
        if model not in KNOWN_MODELS:
            valid = ", ".join(sorted(KNOWN_MODELS))
            raise ValueError(f"Unknown model '{model}'. Expected one of: {valid}")
        if model not in seen:
            unique.append(model)
            seen.add(model)
    return unique


def _weight_grid(models: list[str]) -> list[dict[str, float]]:
    seen = set()
    weights = []
    for raw in itertools.product(WEIGHT_VALUES, repeat=len(models)):
        raw_sum = sum(raw)
        if raw_sum == 0:
            continue
        normalized = tuple(round(value / raw_sum, 10) for value in raw)
        if normalized in seen:
            continue
        seen.add(normalized)
        weights.append(dict(zip(models, normalized)))
    return weights


def _normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"At least one weight must be positive, got {weights}")
    return {model: float(weight / total) for model, weight in weights.items()}


def _fuse_split(
    calibrated_scores: dict[str, dict[str, np.ndarray]],
    split: str,
    weights: dict[str, float],
) -> np.ndarray:
    models = list(weights)
    return fuse(
        [calibrated_scores[model][split] for model in models],
        [weights[model] for model in models],
    )


def _metrics_record(weights: dict[str, float], metrics: dict) -> dict:
    return {
        "weights": {model: float(weight) for model, weight in weights.items()},
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "mca": float(metrics["mca"]),
        "ece": float(metrics["ece"]),
    }


def _sort_grid(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda item: (-item["top1"], -item["mca"], item["ece"]))


def _load_all_scores(
    *,
    score_dir: str,
    dataset_pkl: str,
    models: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, list[str]]]:
    labels_by_split = {}
    frame_dirs_by_split = {}
    for split in SPLIT_TO_DATASET_SPLIT:
        labels, frame_dirs = load_labels(dataset_pkl, split)
        labels_by_split[split] = labels
        frame_dirs_by_split[split] = frame_dirs

    raw_scores = {model: {} for model in models}
    for model in models:
        for split in SPLIT_TO_DATASET_SPLIT:
            scores, score_frame_dirs = _load_scores_with_frame_dirs(score_dir, model, split)
            _assert_score_alignment(
                model=model,
                split=split,
                scores=scores,
                labels=labels_by_split[split],
                frame_dirs=frame_dirs_by_split[split],
                score_frame_dirs=score_frame_dirs,
            )
            raw_scores[model][split] = scores

    temperatures = {}
    calibrated_scores = {model: {} for model in models}
    for model in models:
        temperature = fit_temperature(raw_scores[model]["val"], labels_by_split["val"])
        temperatures[model] = temperature
        for split in SPLIT_TO_DATASET_SPLIT:
            calibrated_scores[model][split] = calibrate(raw_scores[model][split], temperature)

    return temperatures, calibrated_scores, labels_by_split, frame_dirs_by_split


def run(args: argparse.Namespace) -> dict:
    base_models = _unique_models(args.models)
    if not base_models:
        raise ValueError("--models must contain at least one model")
    if args.baseline not in KNOWN_MODELS:
        valid = ", ".join(sorted(KNOWN_MODELS))
        raise ValueError(f"Unknown baseline '{args.baseline}'. Expected one of: {valid}")
    if args.add_a and "a_basic" in base_models:
        raise ValueError("--add-a expects a_basic to be absent from --models")

    models_to_load = _unique_models(
        base_models + [args.baseline] + (["a_basic"] if args.add_a else [])
    )
    temperatures, calibrated_scores, labels_by_split, frame_dirs_by_split = _load_all_scores(
        score_dir=args.score_dir,
        dataset_pkl=args.dataset_pkl,
        models=models_to_load,
    )

    grid_records = []
    for weights in _weight_grid(base_models):
        internal_probs = _fuse_split(calibrated_scores, "internal", weights)
        metrics = evaluate(internal_probs, labels_by_split["internal"])
        grid_records.append(_metrics_record(weights, metrics))
    grid_records = _sort_grid(grid_records)
    selected_internal = grid_records[0]
    selected_weights = selected_internal["weights"]

    selected_final_probs = _fuse_split(calibrated_scores, "final", selected_weights)
    selected_final_metrics = evaluate(selected_final_probs, labels_by_split["final"])
    selected_pred = selected_final_probs.argmax(axis=1)
    confusion_matrix_path = args.confusion_matrix_output
    if confusion_matrix_path is None:
        confusion_matrix_path = Path(args.output).with_name("ensemble_confusion_matrix_final.png")
    confusion_matrix_path = save_confusion_matrix_image(
        selected_pred,
        labels_by_split["final"],
        confusion_matrix_path,
        label_map_file=args.label_map,
        title="Ensemble Final Confusion Matrix",
    )
    baseline_final_probs = calibrated_scores[args.baseline]["final"]
    baseline_pred = baseline_final_probs.argmax(axis=1)
    p_value = mcnemar_exact(selected_pred, baseline_pred, labels_by_split["final"])
    diff = sample_diff(
        selected_pred,
        baseline_pred,
        labels_by_split["final"],
        frame_dirs_by_split["final"],
    )

    report = {
        "temperatures": {model: float(temperatures[model]) for model in models_to_load},
        "grid_internal": grid_records,
        "selected": {
            "weights": selected_weights,
            "internal": {
                key: value
                for key, value in selected_internal.items()
                if key != "weights"
            },
            "final": selected_final_metrics,
            "baseline": args.baseline,
            "confusion_matrix": confusion_matrix_path,
            "mcnemar_p_vs_baseline": float(p_value),
            "diff_vs_baseline": diff,
        },
    }

    if args.add_a:
        report["add_a"] = _evaluate_add_a(
            calibrated_scores=calibrated_scores,
            labels_by_split=labels_by_split,
            selected_weights=selected_weights,
            selected_internal_top1=selected_internal["top1"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    _print_summary(
        report=report,
        output_path=output_path,
        baseline=args.baseline,
    )
    return report


def _evaluate_add_a(
    *,
    calibrated_scores: dict[str, dict[str, np.ndarray]],
    labels_by_split: dict[str, np.ndarray],
    selected_weights: dict[str, float],
    selected_internal_top1: float,
) -> dict:
    candidates = []
    promoted = []
    for add_weight in ADD_A_WEIGHTS:
        raw_weights = dict(selected_weights)
        raw_weights["a_basic"] = add_weight
        normalized_weights = _normalize_weight_dict(raw_weights)

        internal_probs = _fuse_split(calibrated_scores, "internal", normalized_weights)
        internal_metrics = evaluate(internal_probs, labels_by_split["internal"])
        candidate = {
            "raw_weights": {model: float(weight) for model, weight in raw_weights.items()},
            "weights": normalized_weights,
            "internal": internal_metrics,
        }
        candidates.append(candidate)

        if internal_metrics["top1"] > selected_internal_top1:
            final_probs = _fuse_split(calibrated_scores, "final", normalized_weights)
            promoted.append(
                {
                    "raw_weights": candidate["raw_weights"],
                    "weights": normalized_weights,
                    "internal": internal_metrics,
                    "final": evaluate(final_probs, labels_by_split["final"]),
                }
            )

    return {"candidates_internal": candidates, "promoted": promoted}


def _print_summary(*, report: dict, output_path: Path, baseline: str) -> None:
    print("Temperatures:")
    for model, temperature in report["temperatures"].items():
        print(f"  {model}: {temperature:.1f}")

    print("\nInternal top 5:")
    for rank, item in enumerate(report["grid_internal"][:5], start=1):
        print(
            f"  {rank}. weights={_format_weights(item['weights'])} "
            f"top1={item['top1']:.4f} mca={item['mca']:.4f} ece={item['ece']:.4f}"
        )

    selected = report["selected"]
    final = selected["final"]
    print("\nSelected final:")
    print(
        f"  weights={_format_weights(selected['weights'])} "
        f"top1={final['top1']:.4f} top5={final['top5']:.4f} "
        f"mca={final['mca']:.4f} ece={final['ece']:.4f}"
    )
    print(f"  Confusion matrix: {selected['confusion_matrix']}")
    diff = selected["diff_vs_baseline"]
    print(f"  McNemar p vs {baseline}: {selected['mcnemar_p_vs_baseline']:.6g}")
    print(
        f"  Diff counts vs {baseline}: "
        f"selected_only={len(diff['a_only_correct'])}, "
        f"baseline_only={len(diff['b_only_correct'])}"
    )

    if "add_a" in report:
        print("\nAdd-a candidates:")
        for item in report["add_a"]["candidates_internal"]:
            internal = item["internal"]
            print(
                f"  raw={_format_weights(item['raw_weights'])} "
                f"top1={internal['top1']:.4f} mca={internal['mca']:.4f} "
                f"ece={internal['ece']:.4f}"
            )
        print(f"  promoted_final={len(report['add_a']['promoted'])}")

    print(f"\nReport saved: {output_path}")


def _format_weights(weights: dict[str, float]) -> str:
    return "{" + ", ".join(f"{model}:{weight:.4g}" for model, weight in weights.items()) + "}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit val temperatures, select ensemble weights on internal, and evaluate final once."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pt05", "b_T2", "jm_T2w"],
        choices=sorted(KNOWN_MODELS),
        help="Base models used for the internal weight grid.",
    )
    parser.add_argument(
        "--score-dir",
        required=True,
        help="Root directory containing scores/{model}/{split}.pkl files.",
    )
    parser.add_argument(
        "--dataset-pkl",
        required=True,
        help="dataset_diff.pkl path containing split and annotations.",
    )
    parser.add_argument(
        "--baseline",
        default="b_T2",
        choices=sorted(KNOWN_MODELS),
        help="Calibrated single model baseline for final McNemar and diff.",
    )
    parser.add_argument(
        "--output",
        default="results/ensemble_report.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        default=None,
        help=(
            "Path to write selected final confusion matrix PNG. "
            "Defaults to ensemble_confusion_matrix_final.png next to --output."
        ),
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Optional label mapping JSON for confusion matrix tick labels.",
    )
    parser.add_argument(
        "--add-a",
        action="store_true",
        help="Try adding a_basic to the selected ensemble with raw weights 0.2 and 0.3.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (AssertionError, FileNotFoundError, ValueError, KeyError, pickle.PickleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
