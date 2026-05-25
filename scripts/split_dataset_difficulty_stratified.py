import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CONFIGS = [
    ("btc_10s", "meta_btc_10s.csv", "keypoints_btc_10s"),
    ("crawl_10s", "meta_crawl_10s.csv", "keypoints_crawl_10s"),
    ("test", "meta_test.csv", "keypoints_test"),
]

QUALITY_METRICS = [
    "avg_conf",
    "q10_conf",
    "low_conf_joint_ratio",
    "low_conf_frame_ratio",
    "max_low_conf_run",
    "upper_conf",
    "lower_conf",
    "left_conf",
    "right_conf",
    "left_right_conf_gap",
    "bbox_area_mean",
    "bbox_area_cv",
    "bbox_center_motion",
    "zero_coord_ratio",
    "out_of_frame_ratio",
    "motion_energy",
    "velocity_mean",
    "velocity_std",
    "acceleration_mean",
    "jitter_score",
]

DIFFICULTY_COMPONENTS = [
    "low_conf_joint_ratio",
    "low_conf_frame_ratio",
    "jitter_score",
    "out_of_frame_ratio",
    "bbox_area_cv",
]

SPLIT_REPORT_COLUMNS = [
    "sample_id",
    "frame_dir",
    "source_dataset",
    "exercise",
    "file_name",
    "file_stem",
    "split",
    "group_id",
    "label",
    "difficulty_score",
    "difficulty_bin",
    *QUALITY_METRICS,
    "warning",
]

COCO20_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "mid_hip",
    "spine",
    "mid_shoulder",
]

JOINT_INDEX = {name: idx for idx, name in enumerate(COCO20_JOINT_NAMES)}
UPPER_IDX = [
    JOINT_INDEX[name]
    for name in [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "mid_shoulder",
        "spine",
    ]
]
LOWER_IDX = [
    JOINT_INDEX[name]
    for name in [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "mid_hip",
    ]
]
LEFT_IDX = [idx for idx, name in enumerate(COCO20_JOINT_NAMES) if name.startswith("left_")]
RIGHT_IDX = [idx for idx, name in enumerate(COCO20_JOINT_NAMES) if name.startswith("right_")]
TORSO_SCALE_IDX = [
    JOINT_INDEX["left_shoulder"],
    JOINT_INDEX["right_shoulder"],
    JOINT_INDEX["left_hip"],
    JOINT_INDEX["right_hip"],
]

LOW_CONF_THRESHOLD = 0.5
VALID_BBOX_CONFIDENCE = 0.3
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset_v2 from BTC/crawl/test keypoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-pkl", type=str, default="data/datasets/dataset_v2.pkl")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--internal-test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recompute-quality", action="store_true")
    return parser.parse_args()


def append_warning(current, message):
    if not message:
        return current
    if not current:
        return message
    if message in current.split(" | "):
        return current
    return f"{current} | {message}"


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if arr.size else np.nan


def safe_nanquantile(values, q):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else np.nan


def max_true_run(mask):
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def read_metadata(data_dir):
    rows = []
    missing_source_video_id = False
    for source_dataset, meta_name, keypoint_root_name in SOURCE_CONFIGS:
        meta_path = data_dir / meta_name
        if not meta_path.exists():
            raise FileNotFoundError(f"Required metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)
        if "exercise" not in df.columns or "file_name" not in df.columns:
            raise ValueError(f"{meta_path} must contain exercise and file_name columns")

        has_source_video_id = "source_video_id" in df.columns
        if not has_source_video_id:
            missing_source_video_id = True

        for _, row in df.iterrows():
            exercise = str(row["exercise"])
            file_name = str(row["file_name"])
            file_stem = Path(file_name).stem
            if has_source_video_id and pd.notna(row.get("source_video_id")):
                group_key = str(row["source_video_id"])
            else:
                group_key = file_stem

            pickle_path = data_dir / keypoint_root_name / "pickle" / exercise / f"{file_stem}.pkl"
            frame_dir = f"{source_dataset}/{exercise}/{file_stem}"
            rows.append(
                {
                    "sample_id": frame_dir,
                    "frame_dir": frame_dir,
                    "source_dataset": source_dataset,
                    "exercise": exercise,
                    "file_name": file_name,
                    "file_stem": file_stem,
                    "pickle_path": str(pickle_path),
                    "group_id": f"{source_dataset}/{group_key}",
                    "warning": "" if has_source_video_id else "source_video_id_missing",
                }
            )

    if missing_source_video_id:
        warnings.warn("source_video_id column missing; original-video segment leakage cannot be fully guaranteed.")

    meta = pd.DataFrame(rows)
    if meta["frame_dir"].duplicated().any():
        dupes = meta.loc[meta["frame_dir"].duplicated(), "frame_dir"].tolist()
        raise ValueError(f"frame_dir must be unique. Duplicates: {dupes[:10]}")
    return meta


def raw_keypoints_to_coco20(raw_keypoints):
    out = np.zeros((20, 3), dtype=np.float32)
    n_available = min(len(raw_keypoints), 17)
    for idx in range(n_available):
        kp = raw_keypoints[idx]
        out[idx, 0] = safe_float(kp.get("x", 0.0), default=0.0)
        out[idx, 1] = safe_float(kp.get("y", 0.0), default=0.0)
        out[idx, 2] = safe_float(kp.get("confidence", 0.0), default=0.0)

    if n_available >= 17:
        mid_hip = (out[11] + out[12]) / 2.0
        mid_shoulder = (out[5] + out[6]) / 2.0
        spine = (mid_hip + mid_shoulder) / 2.0
        out[17] = mid_hip
        out[18] = spine
        out[19] = mid_shoulder
    elif len(raw_keypoints) >= 20:
        for idx in range(17, 20):
            kp = raw_keypoints[idx]
            out[idx, 0] = safe_float(kp.get("x", 0.0), default=0.0)
            out[idx, 1] = safe_float(kp.get("y", 0.0), default=0.0)
            out[idx, 2] = safe_float(kp.get("confidence", 0.0), default=0.0)
    return out


def stable_total_frames(frames):
    if not frames:
        return 0
    max_idx = max(int(frame.get("frame_idx", idx)) for idx, frame in enumerate(frames))
    return max(len(frames), max_idx + 1)


def load_keypoint_pickle(pickle_path):
    with Path(pickle_path).open("rb") as f:
        frames = pickle.load(f)

    if not isinstance(frames, list):
        raise ValueError(f"Expected list[dict], got {type(frames)!r}")

    total_frames = stable_total_frames(frames)
    keypoint = np.zeros((1, total_frames, 20, 2), dtype=np.float32)
    keypoint_score = np.zeros((1, total_frames, 20), dtype=np.float32)
    bboxes = np.full((total_frames, 4), np.nan, dtype=np.float32)
    pose_confidence = np.full(total_frames, np.nan, dtype=np.float32)
    pose_present = np.zeros(total_frames, dtype=bool)
    unexpected_lengths = set()

    for default_idx, frame in enumerate(frames):
        frame_idx = int(frame.get("frame_idx", default_idx))
        if frame_idx < 0 or frame_idx >= total_frames:
            continue
        poses = frame.get("poses") or []
        if not poses:
            continue

        pose = poses[0]
        raw_keypoints = pose.get("keypoints") or []
        if not raw_keypoints:
            continue
        if len(raw_keypoints) not in (17, 20):
            unexpected_lengths.add(len(raw_keypoints))

        coco20 = raw_keypoints_to_coco20(raw_keypoints)
        keypoint[0, frame_idx] = coco20[:, :2]
        keypoint_score[0, frame_idx] = coco20[:, 2]
        pose_present[frame_idx] = True

        bbox = pose.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            bboxes[frame_idx] = [safe_float(v) for v in bbox[:4]]
        pose_confidence[frame_idx] = safe_float(pose.get("confidence"))

    warning = ""
    if unexpected_lengths:
        warning = append_warning(warning, f"unexpected_keypoint_lengths={sorted(unexpected_lengths)}")
    return {
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
        "total_frames": int(total_frames),
        "bboxes": bboxes,
        "pose_confidence": pose_confidence,
        "pose_present": pose_present,
        "warning": warning,
    }


def frame_confidence_from_joints(confidence, pose_confidence):
    valid_counts = np.isfinite(confidence).sum(axis=1)
    sums = np.nansum(confidence, axis=1)
    joint_frame_conf = np.full(confidence.shape[0], np.nan, dtype=float)
    np.divide(sums, valid_counts, out=joint_frame_conf, where=valid_counts > 0)
    return np.where(np.isfinite(pose_confidence), pose_confidence, joint_frame_conf)


def compute_body_scale(keypoints, bboxes):
    n_frames = keypoints.shape[0]
    scales = np.full(n_frames, np.nan, dtype=float)
    torso = keypoints[:, TORSO_SCALE_IDX, :]
    for t in range(n_frames):
        pts = torso[t]
        valid = np.isfinite(pts).all(axis=1) & ~np.all(np.isclose(pts, 0.0), axis=1)
        if valid.sum() >= 2:
            span = pts[valid].max(axis=0) - pts[valid].min(axis=0)
            diag = float(np.linalg.norm(span))
            if np.isfinite(diag) and diag > EPS:
                scales[t] = diag

    valid_bbox = np.isfinite(bboxes).all(axis=1)
    widths = np.maximum(bboxes[:, 2] - bboxes[:, 0], 0.0)
    heights = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0.0)
    bbox_diag = np.where(valid_bbox, np.hypot(widths, heights), np.nan)
    scales = np.where(np.isfinite(scales) & (scales > EPS), scales, bbox_diag)

    valid_scales = scales[np.isfinite(scales) & (scales > EPS)]
    fallback = float(np.median(valid_scales)) if valid_scales.size else 1.0
    return np.where(np.isfinite(scales) & (scales > EPS), scales, fallback)


def bbox_area_and_center_metrics(bboxes, scales):
    valid_bbox = np.isfinite(bboxes).all(axis=1)
    widths = np.maximum(bboxes[:, 2] - bboxes[:, 0], 0.0)
    heights = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0.0)
    areas = np.where(valid_bbox, widths * heights, np.nan)
    area_mean = safe_nanmean(areas)
    area_std = safe_nanstd(areas)
    area_cv = float(area_std / area_mean) if np.isfinite(area_mean) and area_mean > EPS else np.nan

    centers = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2.0, (bboxes[:, 1] + bboxes[:, 3]) / 2.0))
    valid_centers = valid_bbox & np.isfinite(centers).all(axis=1)
    if len(bboxes) >= 2:
        deltas = centers[1:] - centers[:-1]
        scale_pair = np.maximum((scales[1:] + scales[:-1]) / 2.0, EPS)
        center_motion = np.linalg.norm(deltas / scale_pair[:, None], axis=1)
        center_motion[~(valid_centers[1:] & valid_centers[:-1])] = np.nan
        bbox_center_motion = safe_nanmean(center_motion)
    else:
        bbox_center_motion = np.nan
    return area_mean, area_cv, bbox_center_motion


def coordinate_quality(keypoints):
    finite_coords = np.isfinite(keypoints)
    if finite_coords.any():
        zero_coord_ratio = float(np.isclose(keypoints[finite_coords], 0.0, atol=EPS).mean())
    else:
        zero_coord_ratio = np.nan

    xy_valid = np.isfinite(keypoints).all(axis=2) & ~np.all(np.isclose(keypoints, 0.0), axis=2)
    if not xy_valid.any():
        return zero_coord_ratio, np.nan

    finite_values = keypoints[finite_coords]
    coord_mode = "normalized" if safe_nanquantile(np.abs(finite_values), 0.95) <= 2.0 else "pixel"
    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]
    if coord_mode == "normalized":
        out = (x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)
    else:
        out = (x < 0.0) | (y < 0.0)
    out = out & xy_valid
    return zero_coord_ratio, float(out.sum() / xy_valid.sum())


def normalized_motion_metrics(keypoints, scales):
    valid_points = np.isfinite(keypoints).all(axis=2) & ~np.all(np.isclose(keypoints, 0.0), axis=2)
    if keypoints.shape[0] < 2:
        return {
            "motion_energy": np.nan,
            "velocity_mean": np.nan,
            "velocity_std": np.nan,
            "acceleration_mean": np.nan,
            "jitter_score": np.nan,
        }

    valid_pairs = valid_points[1:] & valid_points[:-1]
    scale_pair = np.maximum((scales[1:] + scales[:-1]) / 2.0, EPS)
    velocity_vec = (keypoints[1:] - keypoints[:-1]) / scale_pair[:, None, None]
    velocity = np.linalg.norm(velocity_vec, axis=2)
    velocity[~valid_pairs] = np.nan

    if keypoints.shape[0] >= 3:
        valid_triples = valid_points[2:] & valid_points[1:-1] & valid_points[:-2]
        scale_triple = np.maximum(scales[1:-1], EPS)
        acceleration_vec = (keypoints[2:] - 2.0 * keypoints[1:-1] + keypoints[:-2]) / scale_triple[:, None, None]
        acceleration = np.linalg.norm(acceleration_vec, axis=2)
        acceleration[~valid_triples] = np.nan
    else:
        acceleration = np.array([], dtype=float)

    return {
        "motion_energy": safe_nanmean(velocity ** 2),
        "velocity_mean": safe_nanmean(velocity),
        "velocity_std": safe_nanstd(velocity),
        "acceleration_mean": safe_nanmean(acceleration),
        "jitter_score": safe_nanquantile(acceleration, 0.90),
    }


def group_confidence(confidence, indices):
    return safe_nanmean(confidence[:, indices])


def compute_quality_from_loaded(sample, loaded):
    keypoints = loaded["keypoint"][0]
    confidence = loaded["keypoint_score"][0]
    bboxes = loaded["bboxes"]
    pose_confidence = loaded["pose_confidence"]
    pose_present = loaded["pose_present"]

    frame_conf = frame_confidence_from_joints(confidence, pose_confidence)
    flat_conf = confidence[np.isfinite(confidence) & (confidence > 0.0)]
    conf_for_low = np.where(np.isfinite(confidence), confidence, 0.0)
    low_joint_mask = conf_for_low < LOW_CONF_THRESHOLD
    low_frame_mask = (~np.isfinite(frame_conf)) | (frame_conf < LOW_CONF_THRESHOLD)
    scales = compute_body_scale(keypoints, bboxes)
    bbox_area_mean, bbox_area_cv, bbox_center_motion = bbox_area_and_center_metrics(bboxes, scales)
    zero_coord_ratio, out_of_frame_ratio = coordinate_quality(keypoints)

    record = {
        "source_dataset": sample["source_dataset"],
        "exercise": sample["exercise"],
        "file_name": sample["file_name"],
        "avg_conf": safe_nanmean(flat_conf),
        "q10_conf": safe_nanquantile(flat_conf, 0.10),
        "low_conf_joint_ratio": float(low_joint_mask.mean()) if low_joint_mask.size else np.nan,
        "low_conf_frame_ratio": float(low_frame_mask.mean()) if low_frame_mask.size else np.nan,
        "max_low_conf_run": max_true_run(low_frame_mask),
        "upper_conf": group_confidence(confidence, UPPER_IDX),
        "lower_conf": group_confidence(confidence, LOWER_IDX),
        "left_conf": group_confidence(confidence, LEFT_IDX),
        "right_conf": group_confidence(confidence, RIGHT_IDX),
        "bbox_area_mean": bbox_area_mean,
        "bbox_area_cv": bbox_area_cv,
        "bbox_center_motion": bbox_center_motion,
        "zero_coord_ratio": zero_coord_ratio,
        "out_of_frame_ratio": out_of_frame_ratio,
        **normalized_motion_metrics(keypoints, scales),
    }
    record["left_right_conf_gap"] = abs(record["left_conf"] - record["right_conf"])
    record["pose_presence_rate"] = float(pose_present.mean()) if len(pose_present) else np.nan
    record["n_frames"] = loaded["total_frames"]
    return record


def minmax_normalize(values, ref_values):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ref_values = pd.to_numeric(ref_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ref_values.empty:
        return pd.Series(0.0, index=values.index)
    lo = float(ref_values.min())
    hi = float(ref_values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def recompute_difficulty(df):
    dev_mask = df["source_dataset"].isin(["btc_10s", "crawl_10s"])
    score = pd.Series(0.0, index=df.index)
    weights = {
        "low_conf_joint_ratio": 0.35,
        "low_conf_frame_ratio": 0.25,
        "jitter_score": 0.20,
        "out_of_frame_ratio": 0.10,
        "bbox_area_cv": 0.10,
    }
    for metric, weight in weights.items():
        score += weight * minmax_normalize(df[metric], df.loc[dev_mask, metric])
    df["difficulty_score"] = score.astype(float)
    return df


def merge_quality(meta, data_dir, recompute_quality):
    quality_path = data_dir / "keypoint_quality_report.csv"
    if quality_path.exists() and not recompute_quality:
        quality = pd.read_csv(quality_path)
        keep_cols = ["source_dataset", "exercise", "file_name", *QUALITY_METRICS]
        quality = quality[[col for col in keep_cols if col in quality.columns]]
        merged = meta.merge(quality, on=["source_dataset", "exercise", "file_name"], how="left")
    else:
        merged = meta.copy()
        for col in QUALITY_METRICS:
            merged[col] = np.nan

    missing_quality = merged[QUALITY_METRICS].isna().all(axis=1)
    if missing_quality.any():
        print(f"Computing quality metrics for {int(missing_quality.sum())} samples without report rows")
        rows = []
        for idx in merged.index[missing_quality]:
            sample = merged.loc[idx].to_dict()
            path = Path(sample["pickle_path"])
            if not path.exists():
                continue
            try:
                loaded = load_keypoint_pickle(path)
                rows.append((idx, compute_quality_from_loaded(sample, loaded)))
            except Exception as exc:
                merged.at[idx, "warning"] = append_warning(merged.at[idx, "warning"], f"quality_error={exc}")
        for idx, values in rows:
            for key, value in values.items():
                if key in QUALITY_METRICS:
                    merged.at[idx, key] = value

    merged = recompute_difficulty(merged)
    return merged


def assign_difficulty_bins(df):
    df = df.copy()
    df["difficulty_bin"] = "final_test"
    dev_mask = df["source_dataset"].isin(["btc_10s", "crawl_10s"])
    for _, group in df[dev_mask].groupby(["source_dataset", "exercise"], sort=False):
        ordered = group.sort_values(["difficulty_score", "frame_dir"], na_position="last").index.to_numpy()
        if len(ordered) == 1:
            df.loc[ordered, "difficulty_bin"] = "mid"
            continue
        chunks = np.array_split(ordered, 3)
        if len(chunks[0]):
            df.loc[chunks[0], "difficulty_bin"] = "easy"
        if len(chunks[1]):
            df.loc[chunks[1], "difficulty_bin"] = "mid"
        if len(chunks[2]):
            df.loc[chunks[2], "difficulty_bin"] = "hard"
    return df


def desired_counts(n, train_ratio, val_ratio, internal_test_ratio):
    if n <= 0:
        return {"train": 0, "val": 0, "internal_test": 0}
    if n == 1:
        return {"train": 1, "val": 0, "internal_test": 0}
    if n == 2:
        return {"train": 1, "val": 1, "internal_test": 0}
    val = max(1, int(round(n * val_ratio)))
    internal = max(1, int(round(n * internal_test_ratio)))
    train = n - val - internal
    if train < 1:
        train = 1
        overflow = train + val + internal - n
        while overflow > 0 and val >= internal and val > 0:
            val -= 1
            overflow -= 1
        while overflow > 0 and internal > 0:
            internal -= 1
            overflow -= 1
    return {"train": train, "val": val, "internal_test": internal}


def split_dev_pool(df, train_ratio, val_ratio, internal_test_ratio, random_seed):
    df = df.copy()
    df["split"] = ""
    rng = np.random.default_rng(random_seed)
    warnings_by_index = {idx: "" for idx in df.index}

    dev_mask = df["source_dataset"].isin(["btc_10s", "crawl_10s"])
    strata = ["source_dataset", "exercise", "difficulty_bin"]
    for stratum_values, group in df[dev_mask].groupby(strata, sort=False):
        group_ids = group["group_id"].drop_duplicates().to_numpy()
        rng.shuffle(group_ids)
        group_sizes = group.groupby("group_id").size().to_dict()
        n_samples = int(len(group))
        counts = desired_counts(n_samples, train_ratio, val_ratio, internal_test_ratio)
        split_counts = {"train": 0, "val": 0, "internal_test": 0}
        group_to_split = {}

        for group_id in group_ids:
            size = group_sizes[group_id]
            deficits = {
                split: counts[split] - split_counts[split]
                for split in ["train", "val", "internal_test"]
            }
            positive = [split for split, deficit in deficits.items() if deficit > 0]
            if positive:
                split = max(positive, key=lambda name: (deficits[name], counts[name]))
            else:
                split = min(split_counts, key=split_counts.get)
            group_to_split[group_id] = split
            split_counts[split] += size

        for group_id, split in group_to_split.items():
            idxs = group.index[group["group_id"] == group_id]
            df.loc[idxs, "split"] = split

        warning = ""
        if n_samples == 1:
            warning = "small_stratum_train_only"
        elif counts["internal_test"] == 0:
            warning = "small_stratum_no_internal_test"
        elif len(group_ids) < 3:
            warning = "small_group_count_fallback"
        if warning:
            for idx in group.index:
                warnings_by_index[idx] = append_warning(warnings_by_index[idx], warning)

    df.loc[df["source_dataset"] == "test", "split"] = "final_test"
    for idx, warning in warnings_by_index.items():
        if warning:
            df.at[idx, "warning"] = append_warning(df.at[idx, "warning"], warning)
    return df


def check_group_leakage(df):
    leakages = []
    dev = df[df["split"].isin(["train", "val", "internal_test"])]
    for group_id, group in dev.groupby("group_id"):
        splits = sorted(group["split"].dropna().unique())
        if len(splits) > 1:
            leakages.append((group_id, splits))
    final_frame_dirs = set(df.loc[df["split"] == "final_test", "frame_dir"])
    dev_frame_dirs = set(df.loc[df["split"].isin(["train", "val", "internal_test"]), "frame_dir"])
    final_overlap = sorted(final_frame_dirs & dev_frame_dirs)
    return leakages, final_overlap


def build_annotations(split_df):
    annotations = []
    skipped = []
    for _, row in split_df.iterrows():
        if row["split"] == "skipped":
            skipped.append(row.to_dict())
            continue
        pickle_path = Path(row["pickle_path"])
        if not pickle_path.exists():
            skipped.append(row.to_dict())
            continue

        loaded = load_keypoint_pickle(pickle_path)
        warning = append_warning(row.get("warning", ""), loaded["warning"])
        annotation = {
            "frame_dir": row["frame_dir"],
            "total_frames": int(loaded["total_frames"]),
            "label": int(row["label"]),
            "keypoint": loaded["keypoint"],
            "keypoint_score": loaded["keypoint_score"],
            "source_dataset": row["source_dataset"],
            "exercise": row["exercise"],
            "file_name": row["file_name"],
            "group_id": row["group_id"],
            "difficulty_score": float(row["difficulty_score"]) if pd.notna(row["difficulty_score"]) else 0.0,
            "difficulty_bin": row["difficulty_bin"],
        }
        if warning:
            annotation["warning"] = warning
        annotations.append(annotation)
    return annotations, skipped


def build_quality_distribution(report_df):
    rows = []
    group_cols = ["split", "source_dataset", "difficulty_bin"]
    metric_cols = [
        "avg_conf",
        "difficulty_score",
        "low_conf_joint_ratio",
        "low_conf_frame_ratio",
        "jitter_score",
        "bbox_area_cv",
    ]
    for keys, group in report_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["count"] = int(len(group))
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean()) if metric in group else np.nan
            row[f"{metric}_std"] = float(group[metric].std(ddof=0)) if metric in group else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_split_summary(report_df):
    print("\nSplit counts:")
    print(report_df["split"].value_counts().to_string())
    print("\nSplit x source_dataset:")
    print(pd.crosstab(report_df["split"], report_df["source_dataset"]).to_string())
    print("\nSplit x difficulty_bin:")
    print(pd.crosstab(report_df["split"], report_df["difficulty_bin"]).to_string())
    print("\nSplit x exercise:")
    print(pd.crosstab(report_df["split"], report_df["exercise"]).to_string())


def validate_ratios(train_ratio, val_ratio, internal_test_ratio):
    total = train_ratio + val_ratio + internal_test_ratio
    if total <= 0:
        raise ValueError("Split ratios must be positive")
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.internal_test_ratio)

    data_dir = Path(args.data_dir)
    output_pkl = Path(args.output_pkl)
    output_dir = Path(args.output_dir)
    datasets_dir = output_pkl.parent
    splits_dir = output_dir / "splits"
    label_mapping_path = datasets_dir / "dataset_v2_label_mapping.json"
    split_report_path = splits_dir / "dataset_v2_split_report.csv"
    quality_distribution_path = splits_dir / "dataset_v2_quality_distribution.csv"

    meta = read_metadata(data_dir)
    exercises = sorted(meta["exercise"].unique())
    label_mapping = {exercise: idx for idx, exercise in enumerate(exercises)}
    meta["label"] = meta["exercise"].map(label_mapping).astype(int)

    split_df = merge_quality(meta, data_dir, args.recompute_quality)
    split_df = assign_difficulty_bins(split_df)

    missing_pkl = ~split_df["pickle_path"].map(lambda p: Path(p).exists())
    if missing_pkl.any():
        split_df.loc[missing_pkl, "warning"] = split_df.loc[missing_pkl, "warning"].map(
            lambda value: append_warning(value, "missing_keypoint_pkl")
        )
        split_df.loc[missing_pkl, "split"] = "skipped"

    available_df = split_df[~missing_pkl].copy()
    available_df = split_dev_pool(
        available_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        internal_test_ratio=args.internal_test_ratio,
        random_seed=args.random_seed,
    )
    split_df.loc[available_df.index, available_df.columns] = available_df

    leakages, final_overlap = check_group_leakage(split_df[split_df["split"] != "skipped"])
    if leakages:
        warnings.warn(f"Group leakage detected for {len(leakages)} groups")
    if final_overlap:
        warnings.warn(f"Final test leakage detected for {len(final_overlap)} frame_dir values")

    full_split_df = split_df.copy()
    report_df = full_split_df.copy()
    for col in SPLIT_REPORT_COLUMNS:
        if col not in report_df.columns:
            report_df[col] = np.nan
    report_df = report_df[SPLIT_REPORT_COLUMNS]
    quality_distribution = build_quality_distribution(report_df[report_df["split"] != "skipped"])
    print_split_summary(report_df)

    print("\nLeakage checks:")
    print(f"  dev group leakage groups: {len(leakages)}")
    print(f"  final_test frame_dir overlap with dev splits: {len(final_overlap)}")

    skipped_rows = report_df[report_df["split"] == "skipped"]
    if not skipped_rows.empty:
        print(f"\nSkipped files: {len(skipped_rows)}")
        print(skipped_rows[["source_dataset", "exercise", "file_name", "warning"]].head(20).to_string(index=False))

    if args.dry_run:
        print("\nDry run: dataset pickle and reports were not written.")
        print(f"Would write dataset: {output_pkl}")
        print(f"Would write label mapping: {label_mapping_path}")
        print(f"Would write split report: {split_report_path}")
        print(f"Would write quality distribution: {quality_distribution_path}")
        return

    annotations, skipped_annotations = build_annotations(full_split_df)
    if skipped_annotations:
        print(f"\nSkipped during annotation build: {len(skipped_annotations)}")

    split = {
        name: report_df.loc[report_df["split"] == name, "frame_dir"].tolist()
        for name in ["train", "val", "internal_test", "final_test"]
    }
    annotation_ids = {ann["frame_dir"] for ann in annotations}
    split = {
        name: [frame_dir for frame_dir in frame_dirs if frame_dir in annotation_ids]
        for name, frame_dirs in split.items()
    }
    dataset = {"split": split, "annotations": annotations}

    datasets_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(dataset, f)
    with label_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    report_df.to_csv(split_report_path, index=False)
    quality_distribution.to_csv(quality_distribution_path, index=False)

    print("\nSaved outputs:")
    print(f"  dataset: {output_pkl}")
    print(f"  label mapping: {label_mapping_path}")
    print(f"  split report: {split_report_path}")
    print(f"  quality distribution: {quality_distribution_path}")


if __name__ == "__main__":
    main()
