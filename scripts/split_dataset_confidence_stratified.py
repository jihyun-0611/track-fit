import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


NUM_COCO_KEYPOINTS = 17
NUM_OUTPUT_KEYPOINTS = 20

SOURCE_CONFIGS = (
    {
        "source_dataset": "btc_10s",
        "meta_csv": "meta_btc_10s.csv",
        "keypoint_dir": "keypoints_btc_10s",
    },
    {
        "source_dataset": "crawl_10s",
        "meta_csv": "meta_crawl_10s.csv",
        "keypoint_dir": "keypoints_crawl_10s",
    },
    {
        "source_dataset": "test",
        "meta_csv": "meta_test.csv",
        "keypoint_dir": "keypoints_test",
    },
)

DEV_SPLITS = ("train", "val", "internal_test")
REPORT_COLUMNS = [
    "sample_id",
    "frame_dir",
    "source_dataset",
    "exercise",
    "file_name",
    "file_stem",
    "split",
    "group_id",
    "label",
    "avg_conf",
    "confidence_difficulty",
    "confidence_bin",
    "total_frames",
    "warning",
]


def safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def average_rows(arr, row_a, row_b):
    return (arr[row_a] + arr[row_b]) / 2.0


def raw_keypoints_to_coco20(raw_keypoints):
    """Convert MediaPipe-mapped COCO17 or legacy COCO20 dicts to Kinetics-style 20 joints."""
    output = np.zeros((NUM_OUTPUT_KEYPOINTS, 3), dtype=np.float32)
    usable = min(len(raw_keypoints or []), NUM_COCO_KEYPOINTS)

    for joint_idx in range(usable):
        kp = raw_keypoints[joint_idx] or {}
        output[joint_idx, 0] = safe_float(kp.get("x"))
        output[joint_idx, 1] = safe_float(kp.get("y"))
        output[joint_idx, 2] = safe_float(kp.get("confidence"))

    if usable >= NUM_COCO_KEYPOINTS:
        mid_hip = average_rows(output, 11, 12)
        mid_shoulder = average_rows(output, 5, 6)
        output[17] = mid_hip
        output[18] = (mid_hip + mid_shoulder) / 2.0
        output[19] = mid_shoulder

    return output


def infer_total_frames(frames):
    if not frames:
        return 0
    max_frame_idx = -1
    for fallback_idx, frame_data in enumerate(frames):
        frame_idx = safe_float(frame_data.get("frame_idx", fallback_idx), fallback_idx)
        max_frame_idx = max(max_frame_idx, int(frame_idx))
    return max(len(frames), max_frame_idx + 1)


def load_keypoint_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    total_frames = infer_total_frames(frames)
    keypoint = np.zeros(
        (1, total_frames, NUM_OUTPUT_KEYPOINTS, 2), dtype=np.float32
    )
    keypoint_score = np.zeros(
        (1, total_frames, NUM_OUTPUT_KEYPOINTS), dtype=np.float32
    )

    for fallback_idx, frame_data in enumerate(frames):
        frame_idx = int(safe_float(frame_data.get("frame_idx", fallback_idx), fallback_idx))
        if frame_idx < 0 or frame_idx >= total_frames:
            continue

        poses = frame_data.get("poses") or []
        if not poses:
            continue

        raw_keypoints = poses[0].get("keypoints") or []
        coco20 = raw_keypoints_to_coco20(raw_keypoints)
        keypoint[0, frame_idx, :, :] = coco20[:, :2]
        keypoint_score[0, frame_idx, :] = coco20[:, 2]

    avg_conf = safe_float(np.nan_to_num(keypoint_score, nan=0.0).mean())
    return keypoint, keypoint_score, total_frames, avg_conf


def add_warning(row, warning):
    if warning:
        row.setdefault("warnings", []).append(warning)


def read_metadata(data_dir):
    frames = []
    for cfg in SOURCE_CONFIGS:
        csv_path = data_dir / cfg["meta_csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Required metadata CSV missing: {csv_path}")

        df = pd.read_csv(csv_path)
        if "exercise" not in df.columns or "file_name" not in df.columns:
            raise ValueError(f"{csv_path} must contain exercise and file_name columns")

        df = df.copy()
        df["source_dataset"] = cfg["source_dataset"]
        df["keypoint_dir"] = cfg["keypoint_dir"]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_label_mapping(meta_df):
    exercises = sorted(str(x) for x in meta_df["exercise"].dropna().unique())
    return {exercise: idx for idx, exercise in enumerate(exercises)}


def build_sample_rows(data_dir, meta_df, label_mapping):
    rows = []
    skipped = []
    missing_source_video_id = "source_video_id" not in meta_df.columns

    for _, meta_row in meta_df.iterrows():
        source_dataset = str(meta_row["source_dataset"])
        exercise = str(meta_row["exercise"])
        file_name = str(meta_row["file_name"])
        file_stem = Path(file_name).stem
        frame_dir = f"{source_dataset}/{exercise}/{file_stem}"
        pkl_path = (
            data_dir
            / str(meta_row["keypoint_dir"])
            / "pickle"
            / exercise
            / f"{file_stem}.pkl"
        )

        if "source_video_id" in meta_df.columns and pd.notna(meta_row.get("source_video_id")):
            source_video_id = str(meta_row["source_video_id"])
            group_id = f"{source_dataset}/{source_video_id}"
        else:
            group_id = f"{source_dataset}/{file_stem}"

        row = {
            "sample_id": frame_dir,
            "frame_dir": frame_dir,
            "source_dataset": source_dataset,
            "exercise": exercise,
            "file_name": file_name,
            "file_stem": file_stem,
            "split": None,
            "group_id": group_id,
            "label": int(label_mapping[exercise]),
            "avg_conf": np.nan,
            "confidence_difficulty": np.nan,
            "confidence_bin": None,
            "total_frames": 0,
            "warning": "",
            "warnings": [],
            "pkl_path": pkl_path,
            "keypoint": None,
            "keypoint_score": None,
        }

        if missing_source_video_id:
            add_warning(row, "source_video_id_missing_sample_split")

        if not pkl_path.exists():
            add_warning(row, f"missing_keypoint_pkl:{pkl_path}")
            row["split"] = "skipped"
            skipped.append(row)
            rows.append(row)
            continue

        keypoint, keypoint_score, total_frames, avg_conf = load_keypoint_pickle(pkl_path)
        row["keypoint"] = keypoint
        row["keypoint_score"] = keypoint_score
        row["total_frames"] = int(total_frames)
        row["avg_conf"] = safe_float(avg_conf)
        row["confidence_difficulty"] = safe_float(1.0 - row["avg_conf"])
        rows.append(row)

    duplicate_frame_dirs = find_duplicates([row["frame_dir"] for row in rows])
    if duplicate_frame_dirs:
        for row in rows:
            if row["frame_dir"] in duplicate_frame_dirs:
                add_warning(row, "duplicate_frame_dir")

    return rows, skipped, missing_source_video_id


def find_duplicates(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def loaded_rows(rows):
    return [row for row in rows if row["keypoint"] is not None]


def assign_confidence_bins(rows):
    for row in rows:
        if row["source_dataset"] == "test":
            row["confidence_bin"] = "final_test"
            row["split"] = "final_test"

    dev_rows = [
        row
        for row in loaded_rows(rows)
        if row["source_dataset"] in {"btc_10s", "crawl_10s"}
    ]
    groups = defaultdict(list)
    for row in dev_rows:
        groups[(row["source_dataset"], row["exercise"])].append(row)

    for _, group_rows in groups.items():
        sorted_rows = sorted(
            group_rows,
            key=lambda row: (-safe_float(row["avg_conf"], -1.0), row["frame_dir"]),
        )
        n_rows = len(sorted_rows)
        for pos, row in enumerate(sorted_rows):
            frac = (pos + 0.5) / max(n_rows, 1)
            if frac <= 1.0 / 3.0:
                row["confidence_bin"] = "easy"
            elif frac <= 2.0 / 3.0:
                row["confidence_bin"] = "mid"
            else:
                row["confidence_bin"] = "hard"


def compute_split_counts(n_samples, train_ratio, val_ratio, internal_test_ratio):
    if n_samples <= 0:
        return {"train": 0, "val": 0, "internal_test": 0}, "empty_stratum"
    if n_samples == 1:
        return {"train": 1, "val": 0, "internal_test": 0}, "small_stratum_train_only"
    if n_samples == 2:
        return {"train": 1, "val": 1, "internal_test": 0}, "small_stratum_no_internal_test"
    if n_samples == 3:
        return {"train": 1, "val": 1, "internal_test": 1}, "small_stratum_even_split"

    n_val = max(1, int(round(n_samples * val_ratio)))
    n_internal = max(1, int(round(n_samples * internal_test_ratio)))
    n_train = n_samples - n_val - n_internal

    while n_train < 1 and (n_val > 1 or n_internal > 1):
        if n_val >= n_internal and n_val > 1:
            n_val -= 1
        elif n_internal > 1:
            n_internal -= 1
        n_train = n_samples - n_val - n_internal

    if n_train < 1:
        n_train = 1
        if n_val >= n_internal and n_val > 0:
            n_val -= 1
        elif n_internal > 0:
            n_internal -= 1

    return {
        "train": int(n_train),
        "val": int(n_val),
        "internal_test": int(n_internal),
    }, None


def assign_group_splits(group_to_rows, target_counts, rng):
    group_ids = list(group_to_rows)
    rng.shuffle(group_ids)

    split_counts = {split: 0 for split in DEV_SPLITS}
    assigned = {}

    for split in ("val", "internal_test"):
        while split_counts[split] < target_counts[split] and group_ids:
            group_id = group_ids.pop(0)
            assigned[group_id] = split
            split_counts[split] += len(group_to_rows[group_id])

    for group_id in group_ids:
        deficits = {
            split: target_counts[split] - split_counts[split]
            for split in DEV_SPLITS
        }
        positive_deficits = {
            split: deficit for split, deficit in deficits.items() if deficit > 0
        }
        if positive_deficits:
            split = max(positive_deficits, key=positive_deficits.get)
        else:
            split = "train"
        assigned[group_id] = split
        split_counts[split] += len(group_to_rows[group_id])

    return assigned


def split_dev_pool(rows, train_ratio, val_ratio, internal_test_ratio, random_seed):
    rng = np.random.default_rng(random_seed)
    strata = defaultdict(list)

    for row in loaded_rows(rows):
        if row["source_dataset"] == "test":
            continue
        stratum = (row["source_dataset"], row["exercise"], row["confidence_bin"])
        strata[stratum].append(row)

    for _, stratum_rows in sorted(strata.items()):
        group_to_rows = defaultdict(list)
        for row in stratum_rows:
            group_to_rows[row["group_id"]].append(row)

        target_counts, fallback_warning = compute_split_counts(
            len(stratum_rows), train_ratio, val_ratio, internal_test_ratio
        )
        if fallback_warning:
            for row in stratum_rows:
                add_warning(row, fallback_warning)

        assignments = assign_group_splits(group_to_rows, target_counts, rng)
        for group_id, split in assignments.items():
            for row in group_to_rows[group_id]:
                row["split"] = split


def check_group_leakage(rows):
    group_splits = defaultdict(set)
    for row in loaded_rows(rows):
        if row["split"] in DEV_SPLITS:
            group_splits[row["group_id"]].add(row["split"])
    return {
        group_id: sorted(splits)
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    }


def check_final_test_isolation(rows):
    bad_rows = []
    for row in loaded_rows(rows):
        if row["source_dataset"] == "test" and row["split"] != "final_test":
            bad_rows.append(row["frame_dir"])
        if row["source_dataset"] != "test" and row["split"] == "final_test":
            bad_rows.append(row["frame_dir"])
    return bad_rows


def finalize_warnings(rows):
    for row in rows:
        row["warning"] = ";".join(sorted(set(row.get("warnings", []))))


def make_dataset(rows):
    annotations = []
    split = {"train": [], "val": [], "internal_test": [], "final_test": []}

    for row in loaded_rows(rows):
        annotation = {
            "frame_dir": row["frame_dir"],
            "total_frames": int(row["total_frames"]),
            "label": int(row["label"]),
            "keypoint": row["keypoint"],
            "keypoint_score": row["keypoint_score"],
            "source_dataset": row["source_dataset"],
            "exercise": row["exercise"],
            "file_name": row["file_name"],
            "group_id": row["group_id"],
            "avg_conf": safe_float(row["avg_conf"]),
            "confidence_difficulty": safe_float(row["confidence_difficulty"]),
            "confidence_bin": row["confidence_bin"],
        }
        annotations.append(annotation)
        if row["split"] in split:
            split[row["split"]].append(row["frame_dir"])

    return {"split": split, "annotations": annotations}


def make_report_df(rows):
    finalize_warnings(rows)
    report_rows = []
    for row in rows:
        report_rows.append({col: row.get(col) for col in REPORT_COLUMNS})
    return pd.DataFrame(report_rows, columns=REPORT_COLUMNS)


def summarize_metric(series):
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def make_quality_distribution(report_df):
    rows = []
    loaded_report = report_df[report_df["split"] != "skipped"].copy()
    group_cols = ["split", "source_dataset", "confidence_bin"]

    for keys, group in loaded_report.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys))
        out["count"] = int(len(group))
        for metric in ("avg_conf", "confidence_difficulty"):
            summary = summarize_metric(group[metric])
            for stat_name, value in summary.items():
                out[f"{metric}_{stat_name}"] = value
        rows.append(out)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def print_split_summary(report_df, rows, skipped, missing_source_video_id):
    loaded_report = report_df[report_df["split"] != "skipped"].copy()
    print("\nSplit counts:")
    print(loaded_report["split"].value_counts().reindex(
        ["train", "val", "internal_test", "final_test"], fill_value=0
    ).to_string())

    print("\nSplit x confidence_bin:")
    print(pd.crosstab(loaded_report["split"], loaded_report["confidence_bin"]).to_string())

    print("\nSplit x source_dataset:")
    print(pd.crosstab(loaded_report["split"], loaded_report["source_dataset"]).to_string())

    print("\nDev confidence_bin coverage:")
    dev = loaded_report[loaded_report["split"].isin(DEV_SPLITS)]
    print(pd.crosstab(dev["split"], dev["confidence_bin"]).to_string())

    final_test_issues = check_final_test_isolation(rows)
    print(f"\nFinal test isolation issues: {len(final_test_issues)}")
    if final_test_issues[:10]:
        print(final_test_issues[:10])

    leakage = check_group_leakage(rows)
    print(f"Group leakage issues across train/val/internal_test: {len(leakage)}")
    if leakage:
        first_items = list(leakage.items())[:10]
        print(first_items)

    if missing_source_video_id:
        print(
            "\nWARNING: source_video_id column is missing. "
            "원본 영상 단위 segment leakage를 완전히 보장할 수 없음; "
            "sample-level group_id was used."
        )

    if skipped:
        print(f"\nSkipped missing keypoint files: {len(skipped)}")
        for row in skipped[:20]:
            print(f"  {row['frame_dir']} -> {row['pkl_path']}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a confidence-only stratified exercise dataset split."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-pkl",
        default="data/datasets/dataset_conf_stratified_v1.pkl",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--internal-test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.internal_test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "train-ratio + val-ratio + internal-test-ratio must equal 1.0"
        )

    data_dir = Path(args.data_dir)
    output_pkl = Path(args.output_pkl)
    output_dir = Path(args.output_dir)
    label_mapping_path = output_pkl.with_name(
        output_pkl.stem + "_label_mapping.json"
    )
    report_path = output_dir / "splits" / "dataset_conf_stratified_v1_report.csv"
    quality_path = (
        output_dir
        / "splits"
        / "dataset_conf_stratified_v1_quality_distribution.csv"
    )

    print(f"Data directory: {data_dir}")
    print("Metadata CSVs: meta_btc_10s.csv, meta_crawl_10s.csv, meta_test.csv")
    print("meta.csv is not read by this script.")

    meta_df = read_metadata(data_dir)
    label_mapping = build_label_mapping(meta_df)
    print(f"Classes: {len(label_mapping)}")

    rows, skipped, missing_source_video_id = build_sample_rows(
        data_dir, meta_df, label_mapping
    )
    assign_confidence_bins(rows)
    split_dev_pool(
        rows,
        args.train_ratio,
        args.val_ratio,
        args.internal_test_ratio,
        args.random_seed,
    )
    report_df = make_report_df(rows)
    quality_df = make_quality_distribution(report_df)
    dataset = make_dataset(rows)

    print_split_summary(report_df, rows, skipped, missing_source_video_id)

    if args.dry_run:
        print("\nDry run: not writing dataset pkl, label mapping, or CSV reports.")
        return

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    label_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(dataset, f)
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    report_df.to_csv(report_path, index=False)
    quality_df.to_csv(quality_path, index=False)

    print("\nWrote:")
    print(f"  dataset: {output_pkl}")
    print(f"  label mapping: {label_mapping_path}")
    print(f"  split report: {report_path}")
    print(f"  quality distribution: {quality_path}")


if __name__ == "__main__":
    main()
