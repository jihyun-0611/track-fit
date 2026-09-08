#!/usr/bin/env python3
"""Run sklearn linear probes over features dumped by tools/extract_features.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protogcn.utils import load_file


SEED = 42
TRAIN_SPLIT = "train"
INTERNAL_SPLIT = "internal_test"
POOLS = ("gap", "vpool")
PROBES: dict[str, tuple[str, ...] | None] = {
    "all22": None,
    "C1": ("bench_press", "incline_bench_press", "decline_bench_press"),
    "C3": ("shoulder_press", "lat_pulldown", "pull_up"),
    "C4": ("deadlift", "romanian_deadlift"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(features_dir: Path) -> dict[str, Any]:
    path = features_dir / "metadata.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata.json must contain a dict: {path}")
    return payload


def discover_hook_blocks(features_dir: Path) -> list[int]:
    blocks = set()
    prefix = f"{TRAIN_SPLIT}_layer"
    suffix = "_gap.npy"
    for path in features_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        layer_text = name[len(prefix) : -len(suffix)]
        try:
            blocks.add(int(layer_text))
        except ValueError:
            continue
    if not blocks:
        raise FileNotFoundError(
            f"No {TRAIN_SPLIT}_layer*_gap.npy files found in {features_dir}; "
            "run tools/extract_features.py first."
        )
    return sorted(blocks)


def load_frame_dirs(features_dir: Path, split: str) -> list[str]:
    path = features_dir / f"{split}_frame_dirs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frame_dir file: {path}")
    frame_dirs = read_json(path)
    if not isinstance(frame_dirs, list) or not all(isinstance(item, str) for item in frame_dirs):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return frame_dirs


def load_labels(features_dir: Path, split: str) -> np.ndarray:
    path = features_dir / f"{split}_labels.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")
    labels = np.load(path)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return labels


def _frame_dir_aliases(frame_dir: str) -> tuple[str, ...]:
    normalized = str(Path(frame_dir).as_posix())
    aliases = [frame_dir, normalized]
    if normalized.startswith("./"):
        aliases.append(normalized[2:])
    return tuple(dict.fromkeys(aliases))


def load_dataset_info(dataset_pkl: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    payload = load_file(str(dataset_pkl))
    if not isinstance(payload, dict) or "split" not in payload or "annotations" not in payload:
        raise ValueError(f"Dataset pkl must contain 'split' and 'annotations': {dataset_pkl}")

    split = payload["split"]
    annotations = payload["annotations"]
    if not isinstance(split, dict) or not isinstance(annotations, list):
        raise ValueError(f"Invalid dataset pkl structure: {dataset_pkl}")

    split_frame_dirs = {
        str(name): [str(frame_dir) for frame_dir in frame_dirs]
        for name, frame_dirs in split.items()
    }

    by_frame_dir: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if "frame_dir" not in annotation or "label" not in annotation or "exercise" not in annotation:
            raise ValueError("Every annotation must contain 'frame_dir', 'label', and 'exercise'")
        frame_dir = str(annotation["frame_dir"])
        if frame_dir in by_frame_dir:
            raise AssertionError(f"Duplicate annotation for frame_dir '{frame_dir}'")
        by_frame_dir[frame_dir] = annotation

    return split_frame_dirs, by_frame_dir


def annotation_for(frame_dir: str, annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for alias in _frame_dir_aliases(frame_dir):
        if alias in annotations:
            return annotations[alias]
    raise KeyError(f"frame_dir '{frame_dir}' is missing from dataset annotations")


def validate_split_alignment(
    *,
    split: str,
    frame_dirs: list[str],
    labels: np.ndarray,
    dataset_splits: dict[str, list[str]],
    annotations: dict[str, dict[str, Any]],
) -> list[str]:
    if split not in dataset_splits:
        raise KeyError(f"Split '{split}' missing from dataset pkl")
    assert frame_dirs == dataset_splits[split], (
        f"Feature frame_dir order for split '{split}' does not match dataset pkl split order"
    )
    assert labels.shape == (len(frame_dirs),), (
        f"Label shape for split '{split}' must be ({len(frame_dirs)},), got {labels.shape}"
    )

    exercises = []
    expected_labels = []
    for frame_dir in frame_dirs:
        annotation = annotation_for(frame_dir, annotations)
        expected_labels.append(int(annotation["label"]))
        exercises.append(str(annotation["exercise"]))
    expected = np.asarray(expected_labels, dtype=np.int64)
    assert np.array_equal(labels, expected), (
        f"Feature labels for split '{split}' do not match dataset annotations"
    )
    return exercises


def load_split_bundle(
    *,
    features_dir: Path,
    split: str,
    dataset_splits: dict[str, list[str]],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    frame_dirs = load_frame_dirs(features_dir, split)
    labels = load_labels(features_dir, split)
    exercises = validate_split_alignment(
        split=split,
        frame_dirs=frame_dirs,
        labels=labels,
        dataset_splits=dataset_splits,
        annotations=annotations,
    )
    return {"frame_dirs": frame_dirs, "labels": labels, "exercises": np.asarray(exercises, dtype=object)}


def load_feature(features_dir: Path, split: str, layer: int, pool: str, n_expected: int) -> np.ndarray:
    path = features_dir / f"{split}_layer{layer}_{pool}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"{path} must contain a 2D array, got shape {features.shape}")
    assert features.shape[0] == n_expected, (
        f"Feature row count for {path} does not match labels: {features.shape[0]} != {n_expected}"
    )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{path} contains NaN or inf")
    return features.astype(np.float32, copy=False)


def probe_mask(exercises: np.ndarray, probe: str) -> np.ndarray:
    allowed = PROBES[probe]
    if allowed is None:
        return np.ones(exercises.shape[0], dtype=bool)
    return np.isin(exercises, np.asarray(allowed, dtype=object))


def make_classifier(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=c,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_and_score(x_train: np.ndarray, y_train: np.ndarray, x_internal: np.ndarray, y_internal: np.ndarray, c: float):
    clf = make_classifier(c)
    clf.fit(x_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(x_train))
    internal_acc = accuracy_score(y_internal, clf.predict(x_internal))
    return float(train_acc), float(internal_acc)


def run_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    ex_train: np.ndarray,
    x_internal: np.ndarray,
    y_internal: np.ndarray,
    ex_internal: np.ndarray,
    probe: str,
) -> dict[str, Any]:
    train_mask = probe_mask(ex_train, probe)
    internal_mask = probe_mask(ex_internal, probe)
    if not train_mask.any():
        raise ValueError(f"Probe {probe} has no train samples")
    if not internal_mask.any():
        raise ValueError(f"Probe {probe} has no internal_test samples")

    x_tr = x_train[train_mask]
    y_tr = y_train[train_mask]
    x_in = x_internal[internal_mask]
    y_in = y_internal[internal_mask]

    classes = np.unique(y_tr)
    if classes.shape[0] < 2:
        raise ValueError(f"Probe {probe} needs at least 2 train classes, got {classes.tolist()}")
    missing_internal = sorted(set(np.unique(y_in).tolist()) - set(classes.tolist()))
    if missing_internal:
        raise ValueError(
            f"Probe {probe} internal_test contains classes absent from train: {missing_internal}"
        )

    train_acc, internal_acc = fit_and_score(x_tr, y_tr, x_in, y_in, c=1.0)
    train_acc_c01, internal_acc_c01 = fit_and_score(x_tr, y_tr, x_in, y_in, c=0.1)
    return {
        "n_train": int(y_tr.shape[0]),
        "n_internal": int(y_in.shape[0]),
        "train_acc": train_acc,
        "internal_acc": internal_acc,
        "train_acc_c01": train_acc_c01,
        "internal_acc_c01": internal_acc_c01,
    }


def format_cell(result: dict[str, Any]) -> str:
    return f"{result['internal_acc']:.3f} (n={result['n_internal']})"


def print_summary(results: list[dict[str, Any]], hook_blocks: list[int], pool: str) -> None:
    probes = list(PROBES)
    by_key = {(item["layer"], item["pool"], item["probe"]): item for item in results}
    headers = ["layer", *probes]
    rows = []
    for layer in hook_blocks:
        row = [str(layer)]
        for probe in probes:
            item = by_key[(layer, pool, probe)]
            row.append(format_cell(item))
        rows.append(row)

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print(f"\n{pool} internal_acc summary")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit sklearn linear probes on extracted ProtoGCN block features."
    )
    parser.add_argument("--features", required=True, help="Feature directory from extract_features.py.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--dataset-pkl",
        default=None,
        help="Dataset pkl path. Defaults to metadata.json's dataset_pkl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    np.random.seed(SEED)

    args = build_arg_parser().parse_args(argv)
    features_dir = Path(args.features).expanduser().resolve()
    if not features_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")

    metadata = load_metadata(features_dir)
    dataset_pkl_value = args.dataset_pkl or metadata.get("dataset_pkl")
    if not dataset_pkl_value:
        raise ValueError("Pass --dataset-pkl or keep metadata.json from extract_features.py")
    dataset_pkl = Path(dataset_pkl_value).expanduser().resolve()
    if not dataset_pkl.exists():
        raise FileNotFoundError(f"Dataset pkl not found: {dataset_pkl}")

    hook_block_values = metadata.get("hook_blocks")
    if hook_block_values is None:
        hook_blocks = discover_hook_blocks(features_dir)
    else:
        hook_blocks = [int(item) for item in hook_block_values]
    model_name = str(metadata.get("model") or features_dir.name)
    dataset_splits, annotations = load_dataset_info(dataset_pkl)

    train_bundle = load_split_bundle(
        features_dir=features_dir,
        split=TRAIN_SPLIT,
        dataset_splits=dataset_splits,
        annotations=annotations,
    )
    internal_bundle = load_split_bundle(
        features_dir=features_dir,
        split=INTERNAL_SPLIT,
        dataset_splits=dataset_splits,
        annotations=annotations,
    )

    results: list[dict[str, Any]] = []
    for layer in hook_blocks:
        for pool in POOLS:
            x_train = load_feature(
                features_dir, TRAIN_SPLIT, layer, pool, len(train_bundle["labels"])
            )
            x_internal = load_feature(
                features_dir, INTERNAL_SPLIT, layer, pool, len(internal_bundle["labels"])
            )
            assert x_train.shape[1] == x_internal.shape[1], (
                f"Feature dimension mismatch for layer{layer}/{pool}: "
                f"{x_train.shape[1]} != {x_internal.shape[1]}"
            )

            for probe in PROBES:
                probe_result = run_probe(
                    x_train=x_train,
                    y_train=train_bundle["labels"],
                    ex_train=train_bundle["exercises"],
                    x_internal=x_internal,
                    y_internal=internal_bundle["labels"],
                    ex_internal=internal_bundle["exercises"],
                    probe=probe,
                )
                results.append(
                    {
                        "layer": int(layer),
                        "pool": pool,
                        "probe": probe,
                        **probe_result,
                    }
                )

    out_payload = {
        "model": model_name,
        "features": str(features_dir),
        "dataset_pkl": str(dataset_pkl),
        "seed": SEED,
        "hook_blocks": hook_blocks,
        "results": results,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"Model: {model_name}")
    print(f"Hook blocks: {hook_blocks}")
    for pool in POOLS:
        print_summary(results, hook_blocks, pool)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
