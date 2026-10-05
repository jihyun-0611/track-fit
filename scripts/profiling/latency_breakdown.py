"""E1: per-segment server latency with the demo input path, inferring every frame.

Segments per frame: base64 decode, imdecode, BGR2RGB, MediaPipe, and once the
buffer holds >= 60 frames: window build, per-stream Compose / forward / .cpu(),
calibrate+fuse. Forward is followed by torch.cuda.synchronize() so GPU time is
not attributed to the .cpu() copy.
"""
import argparse
import base64
import time

import cv2
import numpy as np
import torch

from common import (
    CLIP_LEN,
    NOISY_VIDEOS,
    NORMAL_VIDEOS,
    PROJECT_ROOT,
    WARMUP_VIDEO,
    load_ensemble,
    mark_done,
    percentiles,
    sync,
    write_json,
)
from inference import MIN_FRAMES, calibrate_and_fuse
from protogcn.inference import inference_similarity
from video_input import keypoints_from_rgb, new_pose, read_frames, to_data_url

BUFFER_MAXLEN = 300


def ms_since(start):
    return (time.perf_counter() - start) * 1000.0


def window_from(buffer):
    """Same windowing as DemoInference.predict_rolling_window."""
    frames = list(buffer)
    if len(frames) < CLIP_LEN:
        repeat = (CLIP_LEN // len(frames)) + 1
        frames = (frames * repeat)[:CLIP_LEN]
    else:
        frames = frames[-CLIP_LEN:]
    return np.stack(frames, axis=0)[np.newaxis]


def timed_predict(ensemble, keypoint, seg):
    """CalibratedEnsembleRecognizer.predict with a timer around every segment."""
    probabilities = []
    for index, (model, pipeline) in enumerate(zip(ensemble.models, ensemble.pipelines)):
        t = time.perf_counter()
        data = pipeline({
            "keypoint": keypoint,
            "total_frames": keypoint.shape[1],
            "label": -1,
            "start_index": 0,
            "modality": "Pose",
            "test_mode": True,
        })
        processed = data["keypoint"]
        if not isinstance(processed, torch.Tensor):
            processed = torch.FloatTensor(processed)
        processed = processed.to(ensemble.device)
        sync()
        seg[f"s{index}_compose"] = ms_since(t)

        t = time.perf_counter()
        probs, _ = inference_similarity(model, processed)
        sync()
        seg[f"s{index}_forward"] = ms_since(t)

        t = time.perf_counter()
        probabilities.append(probs.mean(0).cpu().numpy())
        seg[f"s{index}_cpu"] = ms_since(t)

    t = time.perf_counter()
    fused = calibrate_and_fuse(probabilities, ensemble.weights, ensemble.temperatures)[0]
    np.argsort(fused)[::-1]
    seg["calibrate_fuse"] = ms_since(t)


def run_video(ensemble, video_path, target_fps, max_frames=None):
    frames, src_fps, step = read_frames(PROJECT_ROOT / video_path, target_fps)
    if max_frames is not None:
        frames = frames[:max_frames]
    data_urls = [to_data_url(f) for f in frames]  # browser side, not timed
    num_streams = len(ensemble.models)

    records = []
    buffer = []
    with new_pose() as pose:
        for idx, data_url in enumerate(data_urls):
            seg = {}
            t = time.perf_counter()
            img_data = base64.b64decode(data_url.split(",")[1])
            seg["b64decode"] = ms_since(t)

            t = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            seg["imdecode"] = ms_since(t)

            t = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg["bgr2rgb"] = ms_since(t)

            t = time.perf_counter()
            keypoints, detected = keypoints_from_rgb(pose, frame_rgb)
            seg["mediapipe"] = ms_since(t)
            buffer.append(np.asarray(keypoints, dtype=np.float32))
            if len(buffer) > BUFFER_MAXLEN:
                buffer.pop(0)

            inferred = len(buffer) >= MIN_FRAMES
            if inferred:
                t = time.perf_counter()
                keypoint = window_from(buffer)
                seg["window"] = ms_since(t)
                timed_predict(ensemble, keypoint, seg)
                seg["model_total"] = sum(
                    seg[f"s{i}_{part}"] for i in range(num_streams) for part in ("compose", "forward", "cpu")
                )
            seg["server_total"] = sum(v for k, v in seg.items() if k != "model_total")
            records.append({"frame": idx, "pose": detected, "inferred": inferred, "seg": seg})
    return {"video": str(video_path), "src_fps": src_fps, "step": step, "frames": len(frames),
            "records": records}


def summarize(videos):
    """Segment percentiles over inferred frames, excluding each video's first frame."""
    keys = sorted({k for v in videos for r in v["records"] for k in r["seg"]})
    inferred = [r for v in videos for r in v["records"][1:] if r["inferred"]]
    tracking = [r for v in videos for r in v["records"][1:]]
    return {
        "inferred_frames": {k: percentiles([r["seg"][k] for r in inferred if k in r["seg"]]) for k in keys},
        "mediapipe_tracking_all_frames": percentiles([r["seg"]["mediapipe"] for r in tracking]),
        "mediapipe_first_frame": [v["records"][0]["seg"]["mediapipe"] for v in videos if v["records"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--fps", type=int, choices=(10, 30), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    target_fps = 10 if args.fps == 10 else None
    max_frames = 70 if args.smoke else None

    ensemble = load_ensemble()
    run_video(ensemble, WARMUP_VIDEO.relative_to(PROJECT_ROOT), target_fps, max_frames)  # not recorded

    groups = {"normal": NORMAL_VIDEOS, "noisy": NOISY_VIDEOS}
    if args.smoke:
        groups = {"normal": NORMAL_VIDEOS[:1]}
    result = {"meta": {"fps": args.fps, "infer_every_frame_from": MIN_FRAMES,
                       "buffer_maxlen": BUFFER_MAXLEN, "max_frames": max_frames}}
    all_videos = []
    for group, videos in groups.items():
        runs = []
        for video in videos:
            print(f"{group} {video}", flush=True)
            runs.append(run_video(ensemble, video, target_fps, max_frames))
        result[group] = {"videos": runs, "summary": summarize(runs)}
        all_videos.extend(runs)
    result["all_summary"] = summarize(all_videos)

    unit = f"e1_{args.fps}fps"
    write_json(f"{args.out_dir}/{unit}.json", result)
    mark_done(args.out_dir, unit)


if __name__ == "__main__":
    main()
