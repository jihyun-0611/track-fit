"""Reproduce the demo's server-side input path from dataset videos.

Browser side (canvas -> JPEG -> base64 data URL) is done ahead of timing;
the server side (base64 decode -> imdecode -> BGR2RGB -> MediaPipe) is timed.
"""
import base64

import cv2
import mediapipe as mp
import numpy as np

from common import FRAME_SIZE, JPEG_QUALITY
from keypoint_convert import mediapipe_to_coco20


def new_pose():
    """Same settings as demo/app/main.py."""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def read_frames(video_path, target_fps):
    """Read frames, subsampling to roughly target_fps (None keeps the source rate)."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if target_fps is None else max(1, int(round(src_fps / target_fps)))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(cv2.resize(frame, FRAME_SIZE))
        idx += 1
    cap.release()
    return frames, src_fps, step


def to_data_url(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def keypoints_from_rgb(pose, frame_rgb):
    """Run MediaPipe and convert as the demo does; zeros when no pose is found."""
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        return mediapipe_to_coco20(
            results.pose_landmarks, frame_rgb.shape[1], frame_rgb.shape[0]
        ), True
    return np.zeros((20, 3), dtype=np.float32), False


def extract_keypoints(video_path, target_fps=10):
    """Untimed helper: keypoints for every (subsampled) frame of a video."""
    frames, _, _ = read_frames(video_path, target_fps)
    keypoints = []
    with new_pose() as pose:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp, _ = keypoints_from_rgb(pose, rgb)
            keypoints.append(kp)
    return keypoints
