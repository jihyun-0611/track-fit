#!/usr/bin/env python
"""Analyze sample-level prediction reports by quality/source/class groups."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from protogcn.utils.evaluation import top_k_accuracy, mean_class_accuracy, confusion_matrix
from protogcn.utils.metrics import expected_calibration_error


LOGGER = logging.getLogger("analyze_predictions_by_quality")
MISSING_GROUP_VALUE = "__missing__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute overall and grouped metrics from prediction CSV/PKL reports. "
            "The PKL is preferred because it contains score vectors needed for MCA/ECE/MCE."
        )
    )
    parser.add_argument("--predictions-csv", default="reports/internal_test_predictions.csv")
    parser.add_argument("--predictions-pkl", default="reports/internal_test_predictions.pkl")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--prefix", default="internal_test")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--n-bins", type=int, default=15)
    return parser.parse_args()


def normalize_scores(raw_scores: Any) -> np.ndarray | None:
    if raw_scores is None:
        return None
    if isinstance(raw_scores, np.ndarray):
        scores = raw_scores
    elif isinstance(raw_scores, (list, tuple)):
        if not raw_scores:
            return np.empty((0, 0), dtype=np.float32)
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
        raise TypeError(f"Unsupported scores type: {type(raw_scores)!r}")

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, num_classes), got {scores.shape}")
    return scores


def scores_from_dataframe(df: pd.DataFrame) -> np.ndarray | None:
    if "scores" in df.columns:
        try:
            return normalize_scores(df["scores"].tolist())
        except Exception as exc:  # noqa: BLE001 - keep CSV fallback non-fatal
            LOGGER.warning("Could not parse scores column from DataFrame: %s", exc)

    score_columns = [column for column in df.columns if column.startswith("score_")]
    if not score_columns:
        return None

    def score_index(column: str) -> int:
        try:
            return int(column.split("_", 1)[1])
        except ValueError:
            return 10**9

    score_columns = sorted(score_columns, key=score_index)
    return df[score_columns].to_numpy(dtype=np.float32)


def load_predictions(predictions_pkl: Path, predictions_csv: Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    if predictions_pkl.exists():
        with predictions_pkl.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            if "predictions" not in payload:
                raise KeyError(f"Prediction PKL has no 'predictions' key: {predictions_pkl}")
            predictions = payload["predictions"]
            scores = normalize_scores(payload["scores"] if "scores" in payload else None)
        elif isinstance(payload, pd.DataFrame):
            predictions = payload
            scores = scores_from_dataframe(predictions)
        else:
            raise TypeError(f"Unsupported prediction PKL payload type: {type(payload)!r}")
        LOGGER.info("Loaded predictions from PKL: %s", predictions_pkl)
    else:
        if predictions_pkl:
            LOGGER.warning("Prediction PKL not found, falling back to CSV: %s", predictions_pkl)
        if not predictions_csv.exists():
            raise FileNotFoundError(
                f"Neither prediction PKL nor CSV exists: {predictions_pkl}, {predictions_csv}"
            )
        predictions = pd.read_csv(predictions_csv)
        scores = scores_from_dataframe(predictions)
        LOGGER.info("Loaded predictions from CSV: %s", predictions_csv)

    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    predictions = predictions.reset_index(drop=True)
    if scores is not None and len(scores) != len(predictions):
        raise ValueError(
            "Score/prediction length mismatch: "
            f"scores has {len(scores)} rows but predictions has {len(predictions)} rows."
        )
    return predictions, scores


def validate_predictions(df: pd.DataFrame) -> None:
    required = {"label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required prediction columns: {missing}")

    if df["label"].isna().any():
        raise ValueError("Prediction report has missing labels.")


def metric_columns(topk: list[int], include_mca: bool, include_calibration: bool) -> list[str]:
    columns = ["count"] + [f"top{k}" for k in topk]
    if include_mca:
        columns.append("mca")
    if include_calibration:
        columns.extend(["ece", "mce"])
    return columns


def bool_accuracy(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    if series.dtype == bool:
        return float(series.mean())
    normalized = series.astype(str).str.lower()
    return float(normalized.isin(["true", "1", "yes"]).mean())


def compute_metrics(
    df: pd.DataFrame,
    scores: np.ndarray | None,
    indices: np.ndarray,
    topk: list[int],
    n_bins: int,
    group_name: str,
    include_mca: bool = True,
    include_calibration: bool = True,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {"count": int(len(indices))}
    for k in topk:
        row[f"top{k}"] = float("nan")
    if include_mca:
        row["mca"] = float("nan")
    if include_calibration:
        row["ece"] = float("nan")
        row["mce"] = float("nan")

    if len(indices) == 0:
        LOGGER.warning("%s has no samples; metrics set to NaN.", group_name)
        return row

    labels = df.loc[indices, "label"].astype(np.int64).to_numpy()

    if scores is None:
        LOGGER.warning("%s: score vectors unavailable; using correctness columns where possible.", group_name)
        for k in topk:
            correctness_column = f"top{k}_correct"
            if correctness_column in df.columns:
                row[f"top{k}"] = bool_accuracy(df.loc[indices, correctness_column])
        return row

    group_scores = scores[indices]
    try:
        topk_values = top_k_accuracy(group_scores, labels, topk=tuple(topk))
        for k, value in zip(topk, topk_values):
            row[f"top{k}"] = float(value)
    except Exception as exc:  # noqa: BLE001 - keep group analysis robust
        LOGGER.warning("%s: top-k calculation failed: %s", group_name, exc)

    unique_labels = np.unique(labels)
    if include_mca:
        if len(unique_labels) < 2:
            LOGGER.warning(
                "%s has only %d ground-truth class(es); MCA reflects available classes only.",
                group_name,
                len(unique_labels),
            )
        try:
            row["mca"] = float(mean_class_accuracy(group_scores, labels))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s: MCA calculation failed: %s", group_name, exc)

    if include_calibration:
        try:
            calibration = expected_calibration_error(group_scores, labels, n_bins=n_bins)
            row["ece"] = float(calibration["ece"])
            row["mce"] = float(calibration["mce"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s: ECE/MCE calculation failed: %s", group_name, exc)

    return row


def grouped_metrics(
    df: pd.DataFrame,
    scores: np.ndarray | None,
    group_columns: list[str],
    topk: list[int],
    n_bins: int,
    include_mca: bool = True,
    include_calibration: bool = True,
) -> pd.DataFrame:
    missing = [column for column in group_columns if column not in df.columns]
    columns = group_columns + metric_columns(topk, include_mca, include_calibration)
    if missing:
        LOGGER.warning("Missing group column(s) %s; writing empty grouped metrics.", missing)
        return pd.DataFrame(columns=columns)

    rows = []
    group_df = df.copy()
    for column in group_columns:
        group_df[column] = group_df[column].fillna(MISSING_GROUP_VALUE)

    groupby_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in group_df.groupby(groupby_key, dropna=False, sort=True):
        if len(group_columns) == 1:
            keys = (keys,)
        group_name = ", ".join(f"{column}={value}" for column, value in zip(group_columns, keys))
        metrics = compute_metrics(
            df,
            scores,
            group.index.to_numpy(dtype=np.int64),
            topk,
            n_bins,
            group_name=group_name,
            include_mca=include_mca,
            include_calibration=include_calibration,
        )
        rows.append({**dict(zip(group_columns, keys)), **metrics})

    return pd.DataFrame(rows, columns=columns)


def robustness_gap(bin_metrics: pd.DataFrame, bin_column: str) -> float:
    if bin_metrics.empty or bin_column not in bin_metrics.columns or "top1" not in bin_metrics.columns:
        LOGGER.warning("Cannot compute robustness gap for %s; required metrics are missing.", bin_column)
        return float("nan")

    bins = bin_metrics.copy()
    bins["_bin_lower"] = bins[bin_column].astype(str).str.lower()
    easy = bins[bins["_bin_lower"] == "easy"]
    hard = bins[bins["_bin_lower"] == "hard"]
    if easy.empty or hard.empty:
        LOGGER.warning("Cannot compute robustness gap for %s; easy or hard bin is missing.", bin_column)
        return float("nan")
    return float(easy.iloc[0]["top1"] - hard.iloc[0]["top1"])


def save_confusion_matrix(df: pd.DataFrame, scores: np.ndarray | None, output_path: Path) -> None:
    labels = df["label"].astype(np.int64).to_numpy()
    if scores is not None:
        preds = np.argmax(scores, axis=1).astype(np.int64)
    elif "pred_label" in df.columns:
        preds = df["pred_label"].astype(np.int64).to_numpy()
    else:
        LOGGER.warning("Cannot save confusion matrix; pred_label and scores are unavailable.")
        return

    matrix = confusion_matrix(preds, labels)
    np.save(output_path, matrix)
    LOGGER.info("Wrote confusion matrix: %s", output_path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.n_bins < 1:
        raise ValueError(f"--n-bins must be >= 1, got {args.n_bins}")
    if any(k < 1 for k in args.topk):
        raise ValueError(f"--topk values must be >= 1, got {args.topk}")

    topk = sorted(set(args.topk) | {1, 5})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, scores = load_predictions(Path(args.predictions_pkl), Path(args.predictions_csv))
    validate_predictions(df)

    overall_metrics = compute_metrics(
        df,
        scores,
        df.index.to_numpy(dtype=np.int64),
        topk,
        args.n_bins,
        group_name="overall",
        include_mca=True,
        include_calibration=True,
    )
    overall = pd.DataFrame([overall_metrics])

    by_source = grouped_metrics(df, scores, ["source_dataset"], topk, args.n_bins)
    by_exercise = grouped_metrics(
        df,
        scores,
        ["exercise"],
        topk,
        args.n_bins,
        include_mca=False,
        include_calibration=False,
    )
    by_confidence = grouped_metrics(df, scores, ["confidence_bin"], topk, args.n_bins)
    by_difficulty = grouped_metrics(df, scores, ["difficulty_bin"], topk, args.n_bins)
    by_source_confidence = grouped_metrics(
        df,
        scores,
        ["source_dataset", "confidence_bin"],
        topk,
        args.n_bins,
        include_mca=True,
        include_calibration=False,
    )
    by_source_difficulty = grouped_metrics(
        df,
        scores,
        ["source_dataset", "difficulty_bin"],
        topk,
        args.n_bins,
        include_mca=True,
        include_calibration=False,
    )

    overall["robustness_gap_confidence"] = robustness_gap(by_confidence, "confidence_bin")
    overall["robustness_gap_difficulty"] = robustness_gap(by_difficulty, "difficulty_bin")

    prefix = args.prefix
    write_csv(overall, output_dir / f"{prefix}_overall_metrics.csv")
    write_csv(by_source, output_dir / f"{prefix}_by_source.csv")
    write_csv(by_exercise, output_dir / f"{prefix}_by_exercise.csv")
    write_csv(by_confidence, output_dir / f"{prefix}_by_confidence_bin.csv")
    write_csv(by_difficulty, output_dir / f"{prefix}_by_difficulty_bin.csv")
    write_csv(by_source_confidence, output_dir / f"{prefix}_by_source_confidence_bin.csv")
    write_csv(by_source_difficulty, output_dir / f"{prefix}_by_source_difficulty_bin.csv")
    save_confusion_matrix(df, scores, output_dir / f"{prefix}_confusion_matrix.npy")


if __name__ == "__main__":
    main()
