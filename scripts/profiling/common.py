"""Shared helpers for the Track-Fit inference profiling experiment.

All timings synchronize CUDA before reading the clock so that asynchronous
kernel launches are not mistaken for finished work.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "demo"
for path in (PROJECT_ROOT, DEMO_DIR / "extractor", DEMO_DIR / "inferencer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from ensemble_config import load_ensemble_streams  # noqa: E402
from inference import CLIP_LEN, LABEL_MAP, CalibratedEnsembleRecognizer  # noqa: E402

ENSEMBLE_CONFIG_PATH = DEMO_DIR / "configs" / "ensemble.yaml"
WARMUP_VIDEO = PROJECT_ROOT / "data/videos/data_crawl_10s/plank/crawl_plank_82.mp4"
NORMAL_VIDEOS = [
    "data/videos/data_btc_10s/pull_up/pull_up_4.mp4",
    "data/videos/data_btc_10s/bench_press/bench_press_44.mp4",
    "data/videos/data_crawl_10s/lat_pulldown/crawl_lat_pulldown_4.mp4",
    "data/videos/data_btc_10s/russian_twist/russian_twist_6.mp4",
]
NOISY_VIDEOS = [
    "data/videos/data_test/squat/test_squat_2.mp4",
    "data/videos/data_test/hip_thrust/test_hip_thrust_1.mp4",
    "data/videos/data_test/russian_twist/test_russian_twist_1.mp4",
    "data/videos/data_test/decline_bench_press/test_decline_bench_press_3.mp4",
]
FRAME_SIZE = (640, 480)  # (width, height), webcam default assumption
JPEG_QUALITY = 92  # Chrome default for canvas.toBlob('image/jpeg')
RTX_3050TI_LAPTOP_CUDA_CORES = 2560

# (sets, reps per set, eager warmup, compile/graph warmup)
FULL_REPEAT = (3, 200, 50, 100)
SMOKE_REPEAT = (1, 5, 2, 3)


def load_ensemble(stream_names=None, device="cuda:0"):
    """Build the demo ensemble, optionally restricted to the named streams."""
    streams = load_ensemble_streams(
        ENSEMBLE_CONFIG_PATH, PROJECT_ROOT, expected_num_classes=len(LABEL_MAP)
    )
    if stream_names is not None:
        streams = [s for s in streams if s["name"] in stream_names]
        if len(streams) != len(stream_names):
            raise ValueError(f"Unknown stream in {stream_names}")
    return CalibratedEnsembleRecognizer(streams, device=device)


def raw_module(model):
    """Return the module behind a torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    """Run fn once and return (result, elapsed ms) with CUDA synchronized on both ends."""
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter() - start) * 1000.0


def percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_interleaved(configs, repeat, log=print):
    """Time each config in sets, rotating the config order every set.

    configs: dict name -> {"fn": callable, "warmup": int}
    Returns dict name -> {"sets": [[ms, ...], ...], "set_p50": [...], "set_noise": float, "all": stats}
    """
    num_sets, reps, _, _ = repeat
    names = list(configs)
    for name in names:
        log(f"warmup {name} x{configs[name]['warmup']}")
        for _ in range(configs[name]["warmup"]):
            configs[name]["fn"]()
        sync()

    samples = {name: [] for name in names}
    for set_idx in range(num_sets):
        order = names[set_idx % len(names):] + names[:set_idx % len(names)]
        for name in order:
            fn = configs[name]["fn"]
            times = [timed(fn)[1] for _ in range(reps)]
            samples[name].append(times)
            log(f"set {set_idx} {name} p50={np.percentile(times, 50):.3f}ms")

    results = {}
    for name, sets in samples.items():
        set_p50 = [float(np.percentile(s, 50)) for s in sets]
        noise = (max(set_p50) - min(set_p50)) / float(np.median(set_p50)) if set_p50 else None
        results[name] = {
            "sets": sets,
            "set_p50": set_p50,
            "set_noise": noise,
            "all": percentiles([t for s in sets for t in s]),
        }
    return results


def measure_peak_tflops(size=4096, warmup=10, iters=50, trials=3):
    """Measure achievable fp32 matmul throughput (TF32 disabled)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    flops = 2.0 * size ** 3
    best = 0.0
    trial_tflops = []
    for _ in range(warmup):
        a @ b
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            a @ b
        end.record()
        torch.cuda.synchronize()
        tflops = flops * iters / (start.elapsed_time(end) / 1000.0) / 1e12
        trial_tflops.append(tflops)
        best = max(best, tflops)
    torch.backends.cuda.matmul.allow_tf32 = prev
    return {"size": size, "iters": iters, "trial_tflops": trial_tflops, "measured_tflops": best}


def gpu_info():
    query = "name,driver_version,clocks.max.sm,clocks.max.mem,power.limit"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "nvidia_smi": out}
    try:
        max_sm_mhz = float(out.split(",")[2])
        info["spec_fp32_tflops_at_max_sm_clock"] = (
            RTX_3050TI_LAPTOP_CUDA_CORES * 2 * max_sm_mhz * 1e6 / 1e12
        )
    except (IndexError, ValueError):
        info["spec_fp32_tflops_at_max_sm_clock"] = None
    return info


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def mark_done(out_dir, unit):
    (Path(out_dir) / f"{unit}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def demo_input_keypoint(out_dir):
    """Return a (1, 100, 20, 3) keypoint clip from the warmup video at 10 fps.

    Cached in out_dir so every model benchmark sees the same input.
    """
    cache = Path(out_dir) / "input_keypoint.npy"
    if cache.exists():
        return np.load(cache)
    from video_input import extract_keypoints  # local import: needs mediapipe

    frames = extract_keypoints(WARMUP_VIDEO, target_fps=10)
    if len(frames) < CLIP_LEN:
        raise RuntimeError(f"warmup video gave only {len(frames)} frames")
    keypoint = np.stack(frames[-CLIP_LEN:], axis=0)[np.newaxis].astype(np.float32)
    np.save(cache, keypoint)
    return keypoint
