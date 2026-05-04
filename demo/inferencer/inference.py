import torch
import numpy as np
from collections import deque
from protogcn.inference import EnsembleRecognizer


LABEL_MAP = {
    0: "barbell biceps curl",
    1: "bench press",
    2: "chest fly machine",
    3: "deadlift",
    4: "decline bench press",
    5: "hammer curl",
    6: "hip thrust",
    7: "incline bench press",
    8: "lat pulldown",
    9: "lateral raise",
    10: "leg extension",
    11: "leg raises",
    12: "plank",
    13: "pull Up",
    14: "push-up",
    15: "romanian deadlift",
    16: "russian twist",
    17: "shoulder press",
    18: "squat",
    19: "t bar row",
    20: "tricep Pushdown",
    21: "tricep dips",
}

CLIP_LEN = 100
MIN_FRAMES = 60
CONFIDENCE_THRESHOLD = 0.1


class DemoInference:
    """4-stream ensemble inference for real-time exercise recognition.

    Maintains a rolling frame buffer and re-infers using the last CLIP_LEN
    frames on every call to predict_rolling_window().

    Option A ensemble (accuracy-first):
        j  (weight 4) + b  (weight 3) + jm (weight 2) + bm (weight 2)
    """

    def __init__(self, streams, weights=None, device=None):
        """
        Args:
            streams: list of dict, each with 'config' (yaml path) and
                'checkpoint' (pth path).
            weights: per-stream weights (None = equal). Option A: [4, 3, 2, 2].
            device: torch device string. Auto-detected if None.
        """
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.ensemble = EnsembleRecognizer(streams, weights=weights, device=device)
        self.buffer = deque(maxlen=300)
        self.label_map = LABEL_MAP
        self.label_map_reverse = {v: k for k, v in LABEL_MAP.items()}

    def add_frame(self, keypoints):
        """Append one frame to the rolling buffer.

        Args:
            keypoints (np.ndarray): Shape (V, C) = (20, 3).
        """
        self.buffer.append(np.array(keypoints, dtype=np.float32))

    def reset_buffer(self):
        self.buffer.clear()

    def predict_rolling_window(self, selected_exercise=None):  # noqa: ARG002
        """Infer from the last CLIP_LEN frames in the buffer.

        Uses a rolling window: every call takes the most recent CLIP_LEN frames
        and runs all 4 ensemble streams. The fused Option-A-weighted softmax
        scores determine the predicted class.

        Returns:
            dict with 'class', 'confidence', 'all_scores', or None.
        """
        if len(self.buffer) < MIN_FRAMES:
            return None

        frames = list(self.buffer)

        # Repeat-pad when buffer has fewer than CLIP_LEN frames
        if len(frames) < CLIP_LEN:
            repeat = (CLIP_LEN // len(frames)) + 1
            frames = (frames * repeat)[:CLIP_LEN]
        else:
            frames = frames[-CLIP_LEN:]

        # EnsembleRecognizer.predict expects (M, T, V, C)
        keypoint = np.stack(frames, axis=0)[np.newaxis]  # (1, 100, 20, 3)

        # Request all classes for the full score dict
        num_classes = len(self.label_map)
        topk_indices, topk_scores = self.ensemble.predict(keypoint, top_k=num_classes)

        pred_idx = int(topk_indices[0])
        confidence = float(topk_scores[0])

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        all_scores = {
            self.label_map[int(idx)]: float(score)
            for idx, score in zip(topk_indices, topk_scores)
        }

        return {
            "class": self.label_map[pred_idx],
            "confidence": confidence,
            "all_scores": all_scores,
        }
