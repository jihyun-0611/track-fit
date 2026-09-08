#!/usr/bin/env python3
"""Extract linear-probe features from local ProtoGCN backbone blocks."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from tqdm import tqdm

from protogcn.datasets import PoseDataset, build_dataloader
from protogcn.test import build_model_from_cfg, reorder_dataset_to_split_order
from protogcn.utils import load_file


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_name_from_path(path: Path) -> str | None:
    configs_dir = _repo_root() / "configs"
    try:
        rel = path.resolve().relative_to(configs_dir.resolve())
    except ValueError:
        return None
    if rel.suffix not in {".yaml", ".yml"}:
        return None
    return rel.with_suffix("").as_posix()


def load_config(config_arg: str, overrides: list[str]) -> DictConfig:
    """Load a Hydra config name/path, keeping env interpolations lazy."""
    config_path = Path(config_arg).expanduser()
    if config_path.exists():
        config_name = _config_name_from_path(config_path)
        if config_name is None:
            cfg = OmegaConf.load(config_path)
            if overrides:
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
            if "model" not in cfg or "data" not in cfg:
                raise ValueError(
                    f"{config_path} must contain full 'model' and 'data' sections "
                    "or be a Hydra config under configs/."
                )
            return cfg
    else:
        config_name = config_arg[:-5] if config_arg.endswith(".yaml") else config_arg

    configs_dir = _repo_root() / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        try:
            return compose(config_name=config_name, overrides=overrides)
        except Exception:
            has_model_override = any(item.startswith("model=") for item in overrides)
            fallback_model = configs_dir / "model" / "protogcn.yaml"
            if has_model_override or not fallback_model.exists():
                raise
            return compose(config_name=config_name, overrides=["model=protogcn", *overrides])


def load_split_metadata(dataset_pkl: str | Path, split: str) -> tuple[list[str], np.ndarray]:
    payload = load_file(str(dataset_pkl))
    if not isinstance(payload, dict) or "split" not in payload or "annotations" not in payload:
        raise ValueError(f"Dataset pkl must contain 'split' and 'annotations': {dataset_pkl}")
    if split not in payload["split"]:
        raise KeyError(f"Split '{split}' not found in {dataset_pkl}")

    annotations = payload["annotations"]
    if not isinstance(annotations, list):
        raise ValueError(f"Dataset annotations must be a list: {dataset_pkl}")

    by_frame_dir: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if "frame_dir" not in annotation or "label" not in annotation:
            raise ValueError("Every annotation must contain 'frame_dir' and 'label'")
        frame_dir = str(annotation["frame_dir"])
        if frame_dir in by_frame_dir:
            raise AssertionError(f"Duplicate annotation for frame_dir '{frame_dir}'")
        by_frame_dir[frame_dir] = annotation

    frame_dirs = [str(frame_dir) for frame_dir in payload["split"][split]]
    missing = [frame_dir for frame_dir in frame_dirs if frame_dir not in by_frame_dir]
    if missing:
        raise AssertionError(
            f"{len(missing)} frame_dir values from split '{split}' are missing from "
            f"annotations; first missing: {missing[0]}"
        )

    labels = np.asarray([by_frame_dir[frame_dir]["label"] for frame_dir in frame_dirs], dtype=np.int64)
    assert labels.shape == (len(frame_dirs),)
    return frame_dirs, labels


def inference_pipeline_cfg(cfg: DictConfig) -> Any:
    data_cfg = cfg["data"]
    if "test" in data_cfg and "pipeline" in data_cfg["test"]:
        return data_cfg["test"]["pipeline"]
    if "val" in data_cfg and "pipeline" in data_cfg["val"]:
        return data_cfg["val"]["pipeline"]
    raise KeyError("Config must define data.test.pipeline or data.val.pipeline")


def build_dataset(cfg: DictConfig, dataset_pkl: str, split: str) -> PoseDataset:
    data_cfg = cfg["data"]
    dataset = PoseDataset(
        ann_file=dataset_pkl,
        pipeline=inference_pipeline_cfg(cfg),
        split=split,
        data_prefix=data_cfg.get("data_prefix", ""),
        test_mode=True,
    )
    reorder_dataset_to_split_order(
        dataset,
        ann_file=dataset_pkl,
        dataset_split=split,
        data_prefix=data_cfg.get("data_prefix", ""),
    )
    return dataset


def select_hook_blocks(block_count: int) -> list[int]:
    if block_count <= 0:
        raise ValueError("ProtoGCN backbone has no GCN blocks")
    candidates = [0, block_count // 3, (2 * block_count) // 3, block_count - 1]
    selected: list[int] = []
    for index in candidates:
        if index not in selected:
            selected.append(index)
    return selected


def get_gcn_blocks(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, list[int]]:
    raw_model = getattr(model, "_orig_mod", model)
    if not hasattr(raw_model, "backbone") or not hasattr(raw_model.backbone, "gcn"):
        raise AttributeError("Expected model.backbone.gcn to contain ProtoGCN GCN blocks")
    blocks = raw_model.backbone.gcn
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError(f"Expected model.backbone.gcn to be ModuleList, got {type(blocks)}")
    return blocks, select_hook_blocks(len(blocks))


def _hook_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Hook output tuple/list is empty")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected hook output tensor, got {type(output)}")
    return output


def pool_features(
    block_output: np.ndarray,
    *,
    batch_size: int,
    num_clips: int,
    num_person: int,
) -> dict[str, np.ndarray]:
    """Return sample-level GAP and vertex-pool features from one block output."""
    x = np.asarray(block_output)
    if x.ndim == 4:
        expected = batch_size * num_clips * num_person
        if x.shape[0] == expected:
            x = x.reshape(batch_size, num_clips, num_person, *x.shape[1:])
        elif x.shape[0] == batch_size * num_clips:
            x = x.reshape(batch_size, num_clips, 1, *x.shape[1:])
        elif x.shape[0] == batch_size:
            x = x.reshape(batch_size, 1, 1, *x.shape[1:])
        else:
            raise AssertionError(
                f"Cannot map hook output shape {block_output.shape} to "
                f"B={batch_size}, clips={num_clips}, persons={num_person}"
            )
    elif x.ndim == 5:
        if x.shape[0] == batch_size * num_clips and x.shape[1] == num_person:
            x = x.reshape(batch_size, num_clips, num_person, *x.shape[2:])
        elif x.shape[0] == batch_size and x.shape[1] == num_person:
            x = x.reshape(batch_size, 1, num_person, *x.shape[2:])
        else:
            raise AssertionError(
                f"Cannot map hook output shape {block_output.shape} to "
                f"B={batch_size}, clips={num_clips}, persons={num_person}"
            )
    else:
        raise AssertionError(f"Hook output must have 4 or 5 dims, got {block_output.shape}")

    # x: (B, num_clips, M, C, T, V). Average clips/persons to one row per sample.
    gap = x.mean(axis=(1, 2, 4, 5), dtype=np.float64).astype(np.float32)
    vpool = x.mean(axis=(1, 2, 4), dtype=np.float64).astype(np.float32)
    vpool = vpool.reshape(batch_size, -1)
    return {"gap": gap, "vpool": vpool}


def batch_labels(data: dict[str, Any]) -> np.ndarray:
    label = data["label"]
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    label = np.asarray(label)
    if label.ndim > 1:
        label = label.reshape(label.shape[0], -1)[:, 0]
    return label.astype(np.int64).reshape(-1)


@torch.no_grad()
def extract_split(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    hook_blocks: list[int],
    out_dir: Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    chunks: dict[tuple[int, str], list[np.ndarray]] = {
        (block_index, pool): [] for block_index in hook_blocks for pool in ("gap", "vpool")
    }
    observed_labels: list[np.ndarray] = []
    hook_outputs: dict[int, np.ndarray] = {}

    def make_hook(block_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _hook_tensor(output).detach()
            hook_outputs[block_index] = tensor.float().cpu().numpy()

        return hook

    raw_model = getattr(model, "_orig_mod", model)
    handles = [
        raw_model.backbone.gcn[block_index].register_forward_hook(make_hook(block_index))
        for block_index in hook_blocks
    ]
    try:
        for data in tqdm(loader, desc=f"Extract {split}"):
            keypoint = data["keypoint"]
            if not isinstance(keypoint, torch.Tensor):
                keypoint = torch.as_tensor(keypoint)
            if keypoint.ndim != 6:
                raise AssertionError(f"Expected keypoint shape (B, clips, M, T, V, C), got {keypoint.shape}")

            batch_size, num_clips, num_person = map(int, keypoint.shape[:3])
            observed_labels.append(batch_labels(data))
            hook_outputs.clear()

            _ = model(keypoint.to(device, non_blocking=True).float(), return_loss=False)

            missing = [block_index for block_index in hook_blocks if block_index not in hook_outputs]
            if missing:
                raise AssertionError(f"Missing hook outputs for blocks: {missing}")

            for block_index in hook_blocks:
                pooled = pool_features(
                    hook_outputs[block_index],
                    batch_size=batch_size,
                    num_clips=num_clips,
                    num_person=num_person,
                )
                for pool_name, array in pooled.items():
                    chunks[(block_index, pool_name)].append(array)

            hook_outputs.clear()
    finally:
        for handle in handles:
            handle.remove()

    observed = np.concatenate(observed_labels, axis=0) if observed_labels else np.empty((0,), dtype=np.int64)
    assert observed.shape == labels.shape, (
        f"Label count mismatch for split '{split}': observed={observed.shape}, expected={labels.shape}"
    )
    assert np.array_equal(observed, labels), (
        f"Dataloader label order does not match dataset pkl split order for '{split}'"
    )

    written: dict[str, Any] = {"sample_count": int(labels.shape[0]), "features": {}}
    for block_index in hook_blocks:
        written["features"][str(block_index)] = {}
        for pool_name in ("gap", "vpool"):
            features = np.concatenate(chunks[(block_index, pool_name)], axis=0)
            assert features.shape[0] == labels.shape[0], (
                f"Feature row count mismatch for {split}/layer{block_index}/{pool_name}: "
                f"{features.shape[0]} != {labels.shape[0]}"
            )
            feature_path = out_dir / f"{split}_layer{block_index}_{pool_name}.npy"
            np.save(feature_path, features.astype(np.float32, copy=False))
            written["features"][str(block_index)][pool_name] = {
                "path": feature_path.name,
                "shape": list(features.shape),
            }

    return written


def write_split_metadata(out_dir: Path, split: str, frame_dirs: list[str], labels: np.ndarray) -> None:
    frame_path = out_dir / f"{split}_frame_dirs.json"
    label_path = out_dir / f"{split}_labels.npy"
    frame_path.write_text(json.dumps(frame_dirs, indent=2), encoding="utf-8")
    np.save(label_path, labels.astype(np.int64, copy=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GAP and vertex-pooled features from ProtoGCN GCN blocks."
    )
    parser.add_argument("--config", required=True, help="Hydra config name or YAML path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path.")
    parser.add_argument("--dataset-pkl", required=True, help="dataset_diff.pkl path.")
    parser.add_argument("--splits", nargs="+", default=["train", "internal_test"])
    parser.add_argument("--out", required=True, help="Output feature directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    set_seed(SEED)

    parser = build_arg_parser()
    args, overrides = parser.parse_known_args(argv)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset_pkl = Path(args.dataset_pkl).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_pkl.exists():
        raise FileNotFoundError(f"Dataset pkl not found: {dataset_pkl}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but CUDA is not available")

    cfg = load_config(args.config, overrides)
    model = build_model_from_cfg(cfg, str(checkpoint), device=str(device))
    model.eval()

    blocks, hook_blocks = get_gcn_blocks(model)
    hook_paths = [f"model.backbone.gcn.{index}" for index in hook_blocks]
    print(f"GCN blocks: {len(blocks)}")
    print(f"Selected hook blocks: {hook_blocks}")
    print("Hook module paths:")
    for path in hook_paths:
        print(f"  {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or out_dir.name or checkpoint.parent.name
    metadata: dict[str, Any] = {
        "model": model_name,
        "config": args.config,
        "config_overrides": overrides,
        "checkpoint": str(checkpoint),
        "dataset_pkl": str(dataset_pkl),
        "seed": SEED,
        "block_count": len(blocks),
        "hook_blocks": hook_blocks,
        "hook_paths": hook_paths,
        "pools": ["gap", "vpool"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    data_cfg = cfg["data"]
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("workers_per_gpu", 4))
    pin_memory = device.type == "cuda"

    for split in args.splits:
        frame_dirs, labels = load_split_metadata(dataset_pkl, split)
        dataset = build_dataset(cfg, str(dataset_pkl), split)
        dataset_labels = np.asarray([int(item["label"]) for item in dataset.video_infos], dtype=np.int64)
        assert np.array_equal(dataset_labels, labels), (
            f"Dataset labels for split '{split}' do not match dataset pkl split order"
        )
        write_split_metadata(out_dir, split, frame_dirs, labels)

        loader = build_dataloader(
            dataset,
            batch_size=int(args.batch_size),
            num_workers=num_workers,
            shuffle=False,
            seed=SEED,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        split_meta = extract_split(
            model=model,
            loader=loader,
            labels=labels,
            hook_blocks=hook_blocks,
            out_dir=out_dir,
            split=split,
            device=device,
        )
        split_meta["labels"] = f"{split}_labels.npy"
        split_meta["frame_dirs"] = f"{split}_frame_dirs.json"
        metadata["splits"][split] = split_meta

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
