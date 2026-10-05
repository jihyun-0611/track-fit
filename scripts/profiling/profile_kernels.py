"""E4: kernel count, GPU busy ratio, top kernels and achieved FLOP rate.

Profiles the demo predict() path as-is for the 1-stream (j) and 3-stream ensembles.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.flop_counter import FlopCounterMode

from bench_model import J_STREAM, preprocess
from common import (
    FULL_REPEAT,
    SMOKE_REPEAT,
    demo_input_keypoint,
    load_ensemble,
    mark_done,
    run_interleaved,
    write_json,
)
from protogcn.inference import inference_similarity


def merged_length(intervals):
    """Total length of the union of (start, end) intervals."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def analyze_trace(trace_path):
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    # CPU-side step markers only; the trace also carries gpu_user_annotation copies.
    steps = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
             and str(e.get("name", "")).startswith("ProfilerStep#")]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    gpu_work = [e for e in events if e.get("ph") == "X"
                and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    per_step = []
    for step in steps:
        s0, s1 = step["ts"], step["ts"] + step["dur"]
        busy = merged_length([(g["ts"], g["ts"] + g["dur"]) for g in gpu_work if s0 <= g["ts"] < s1])
        per_step.append({
            "wall_us": step["dur"],
            "kernels": sum(1 for k in kernels if s0 <= k["ts"] < s1),
            "gpu_busy_us": busy,
            "gpu_busy_ratio": busy / step["dur"] if step["dur"] else None,
        })
    by_name = defaultdict(lambda: [0.0, 0])
    for k in kernels:
        by_name[k["name"]][0] += k["dur"]
        by_name[k["name"]][1] += 1
    total_kernel_us = sum(v[0] for v in by_name.values())
    top5 = sorted(by_name.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
    return {
        "num_steps": len(steps),
        "num_kernel_events": len(kernels),
        "per_step": per_step,
        "top5_kernels": [
            {"name": name, "total_us": us, "calls": calls,
             "share_of_kernel_time": us / total_kernel_us if total_kernel_us else None}
            for name, (us, calls) in top5
        ],
    }


def profile_predict(ensemble, keypoint, trace_path, active):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=5, active=active, repeat=1),
    ) as prof:
        for _ in range(1 + 5 + active):
            ensemble.predict(keypoint)
            prof.step()
    prof.export_chrome_trace(str(trace_path))
    return analyze_trace(trace_path)


def count_flops(ensemble, keypoint):
    per_stream = {}
    for index, model in enumerate(ensemble.models):
        x = preprocess(ensemble, index, keypoint)
        with FlopCounterMode(display=False) as counter:
            inference_similarity(model, x)
        per_stream[f"stream{index}"] = counter.get_total_flops()
    return per_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repeat = SMOKE_REPEAT if args.smoke else FULL_REPEAT
    active = repeat[1] if args.smoke else 20

    keypoint = demo_input_keypoint(args.out_dir)
    ensembles = {"1stream": load_ensemble([J_STREAM]), "3stream": load_ensemble()}
    payload = {"meta": {"profiled_steps": active, "stream_names": {
        k: [J_STREAM] if k == "1stream" else "all" for k in ensembles}}}

    # Forward-only timing for the achieved FLOP rate (model forward, synchronized).
    configs = {}
    for name, ens in ensembles.items():
        inputs = [preprocess(ens, i, keypoint) for i in range(len(ens.models))]

        def forward_all(ens=ens, inputs=inputs):
            for model, x in zip(ens.models, inputs):
                inference_similarity(model, x)

        configs[f"forward_{name}"] = {"fn": forward_all, "warmup": repeat[2]}
    payload["forward_timing"] = run_interleaved(configs, repeat, log=lambda m: print(m, flush=True))

    for name, ens in ensembles.items():
        flops = count_flops(ens, keypoint)
        trace_path = Path(args.out_dir) / f"e4_trace_{name}.json"
        analysis = profile_predict(ens, keypoint, trace_path, active)
        print(name, "kernel events:", analysis["num_kernel_events"], flush=True)
        payload[name] = {"flops_per_stream": flops, "total_flops": sum(flops.values()), "profile": analysis}

    write_json(f"{args.out_dir}/e4.json", payload)
    mark_done(args.out_dir, "e4")


if __name__ == "__main__":
    main()
