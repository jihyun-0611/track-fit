#!/usr/bin/env python
"""Build a sample-level prediction report from saved ProtoGCN scores."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("save_predictions_from_scores")

OUTPUT_COLUMNS = [
    "frame_dir",
    "split",
    "source_dataset",
    "exercise",
    "label",
    "pred_label",
    "pred_exercise",
    "pred_score",
    "top1_correct",
    "top5_correct",
    "confidence_bin",
    "difficulty_bin",
    "avg_conf",
    "confidence_difficulty",
    "difficulty_score",
    "file_name",
    "group_id",
]

METADATA_COLUMNS = [
    "source_dataset",
    "exercise",
    "label",
    "confidence_bin",
    "difficulty_bin",
    "avg_conf",
    "confidence_difficulty",
    "difficulty_score",
    "file_name",
    "group_id",
]

PREDICTION_COLUMNS = {
    "pred_label",
    "pred_exercise",
    "pred_score",
    "top1_correct",
    "top5_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sample-level prediction CSV/PKL by joining saved scores "
            "with a dataset pickle and optional split report."
        )
    )
    parser.add_argument("--dataset-pkl", default="data/datasets/dataset_diff.pkl")
    parser.add_argument("--split", default="internal_test")
    parser.add_argument("--scores-pkl", default="work_dirs/temporal_occlusion_100/best_internal_test_pred.pkl")
    parser.add_argument("--split-report", default="data/splits/dataset_diff_split_report.csv")
    parser.add_argument("--label-mapping", default="data/datasets/dataset_diff_label_mapping.json")
    parser.add_argument("--output-csv", default="work_dirs/temporal_occlusion_100/internal_test_predictions.csv")
    parser.add_argument("--output-pkl", default="work_dirs/temporal_occlusion_100/internal_test_predictions.pkl")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_scores(raw_scores: Any) -> np.ndarray:
    """Convert saved score payloads to a 2D numpy array."""
    if isinstance(raw_scores, dict):
        for key in ("scores", "all_scores", "predictions"):
            if key in raw_scores:
                raw_scores = raw_scores[key]
                break

    if isinstance(raw_scores, np.ndarray):
        scores = raw_scores
    elif isinstance(raw_scores, (list, tuple)):
        if not raw_scores:
            raise ValueError("scores-pkl is an empty list")
        arrays = [np.asarray(item) for item in raw_scores]
        if all(arr.ndim == 1 for arr in arrays):
            scores = np.stack(arrays, axis=0)
        elif all(arr.ndim == 2 for arr in arrays):
            scores = np.concatenate(arrays, axis=0)
        else:
            squeezed = [np.squeeze(arr) for arr in arrays]
            if all(arr.ndim == 1 for arr in squeezed):
                scores = np.stack(squeezed, axis=0)
            elif all(arr.ndim == 2 for arr in squeezed):
                scores = np.concatenate(squeezed, axis=0)
            else:
                shapes = [arr.shape for arr in arrays[:5]]
                raise ValueError(
                    "Could not normalize score list to shape (N, num_classes); "
                    f"first shapes: {shapes}"
                )
    else:
        raise TypeError(
            "scores-pkl must contain list[np.ndarray] or np.ndarray, "
            f"but got {type(raw_scores)!r}"
        )

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, num_classes), got {scores.shape}")
    return scores


def warn_if_scores_do_not_look_like_probabilities(scores: np.ndarray) -> None:
    if scores.size == 0:
        return
    if np.nanmin(scores) < -1e-6 or np.nanmax(scores) > 1.0 + 1e-6:
        LOGGER.warning("Scores contain values outside [0, 1]; ECE later assumes probabilities.")
        return
    row_sums = scores.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        LOGGER.warning("Score rows do not sum to 1 within tolerance; ECE later assumes probabilities.")


def annotation_key(annotation: dict[str, Any]) -> Any:
    return annotation.get("frame_dir", annotation.get("filename"))


def load_samples(dataset: dict[str, Any], split: str) -> pd.DataFrame:
    if "split" not in dataset or "annotations" not in dataset:
        raise KeyError("dataset pkl must contain 'split' and 'annotations' keys")
    if split not in dataset["split"]:
        available = ", ".join(sorted(dataset["split"].keys()))
        raise KeyError(f"Split '{split}' not found in dataset. Available splits: {available}")

    split_frame_dirs = list(dataset["split"][split])
    annotations = dataset["annotations"]
    by_frame_dir: dict[Any, dict[str, Any]] = {}
    duplicate_count = 0

    for annotation in annotations:
        key = annotation_key(annotation)
        if key is None:
            continue
        if key in by_frame_dir:
            duplicate_count += 1
            continue
        by_frame_dir[key] = annotation

    if duplicate_count:
        LOGGER.warning("Ignored %d duplicate annotation frame_dir entries.", duplicate_count)

    rows = []
    missing = []
    for frame_dir in split_frame_dirs:
        annotation = by_frame_dir.get(frame_dir)
        row = {"frame_dir": frame_dir, "split": split}
        if annotation is None:
            missing.append(frame_dir)
        else:
            for column in METADATA_COLUMNS:
                row[column] = annotation.get(column)
        rows.append(row)

    if missing:
        LOGGER.warning(
            "%d split samples were not found in annotations; first missing frame_dir=%s",
            len(missing),
            missing[0],
        )

    df = pd.DataFrame(rows)
    if "label" not in df.columns or df["label"].isna().any():
        missing_count = int(df["label"].isna().sum()) if "label" in df.columns else len(df)
        raise ValueError(f"Missing label for {missing_count} split samples.")
    return df


def load_label_mapping(path: Path) -> dict[int, str]:
    if not path.exists():
        LOGGER.warning("Label mapping not found: %s", path)
        return {}

    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    idx_to_name = {}
    for key, value in mapping.items():
        try:
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                idx_to_name[int(value)] = str(key)
            else:
                idx_to_name[int(key)] = str(value)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping unsupported label mapping entry: %r -> %r", key, value)
    return idx_to_name


def add_prediction_columns(
    df: pd.DataFrame,
    scores: np.ndarray,
    idx_to_name: dict[int, str],
    topk: int,
) -> pd.DataFrame:
    if len(df) != len(scores):
        raise ValueError(
            "Score/sample length mismatch: "
            f"scores has {len(scores)} rows but split has {len(df)} samples."
        )
    if topk < 1:
        raise ValueError(f"--topk must be >= 1, got {topk}")

    labels = df["label"].astype(np.int64).to_numpy()
    pred_label = np.argmax(scores, axis=1).astype(np.int64)
    pred_score = np.max(scores, axis=1)
    effective_topk = min(topk, scores.shape[1])
    topk_pred = np.argsort(scores, axis=1)[:, -effective_topk:][:, ::-1]
    topk_correct = (topk_pred == labels[:, None]).any(axis=1)

    result = df.copy()
    result["pred_label"] = pred_label
    result["pred_exercise"] = [idx_to_name.get(int(label), pd.NA) for label in pred_label]
    result["pred_score"] = pred_score.astype(float)
    result["top1_correct"] = pred_label == labels
    result["top5_correct"] = topk_correct
    return result


def merge_split_report(df: pd.DataFrame, split_report: Path, split: str) -> pd.DataFrame:
    if not split_report.exists():
        LOGGER.warning("Split report not found: %s", split_report)
        return df

    report = pd.read_csv(split_report)
    if "frame_dir" not in report.columns:
        LOGGER.warning("Split report has no frame_dir column; skipping merge: %s", split_report)
        return df

    if "split" in report.columns:
        report_for_split = report[report["split"].astype(str) == split]
        if not report_for_split.empty:
            report = report_for_split

    merge_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in PREDICTION_COLUMNS and column not in {"frame_dir", "split"}
    ]
    missing_columns = [column for column in merge_columns if column not in report.columns]
    for column in missing_columns:
        LOGGER.warning("Split report is missing column '%s'; continuing.", column)

    available_columns = ["frame_dir"] + [column for column in merge_columns if column in report.columns]
    report = report[available_columns].drop_duplicates(subset=["frame_dir"], keep="first")

    merged = df.merge(report, on="frame_dir", how="left", suffixes=("", "_report"))
    for column in merge_columns:
        report_column = f"{column}_report"
        if report_column not in merged.columns:
            continue
        if column in merged.columns:
            if merged[column].notna().any():
                merged[column] = merged[column].where(merged[column].notna(), merged[report_column])
            else:
                merged[column] = merged[report_column]
        else:
            merged[column] = merged[report_column]
        merged = merged.drop(columns=[report_column])

    return merged


def save_outputs(
    df: pd.DataFrame,
    scores: np.ndarray,
    args: argparse.Namespace,
) -> None:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    output_df = df[OUTPUT_COLUMNS].copy()

    output_csv = Path(args.output_csv)
    output_pkl = Path(args.output_pkl)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_csv, index=False)
    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "predictions": output_df,
                "scores": scores,
                "split": args.split,
                "topk": args.topk,
                "dataset_pkl": args.dataset_pkl,
                "scores_pkl": args.scores_pkl,
                "split_report": args.split_report,
                "label_mapping": args.label_mapping,
            },
            f,
        )

    LOGGER.info("Wrote CSV: %s", output_csv)
    LOGGER.info("Wrote PKL: %s", output_pkl)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    dataset = load_pickle(Path(args.dataset_pkl))
    df = load_samples(dataset, args.split)

    scores = normalize_scores(load_pickle(Path(args.scores_pkl)))
    warn_if_scores_do_not_look_like_probabilities(scores)

    idx_to_name = load_label_mapping(Path(args.label_mapping))
    df = add_prediction_columns(df, scores, idx_to_name, args.topk)
    df = merge_split_report(df, Path(args.split_report), args.split)
    save_outputs(df, scores, args)


if __name__ == "__main__":
    main()
