import torch
import numpy as np
from collections import deque
from omegaconf import OmegaConf

from protogcn.datasets.pipelines import Compose
from protogcn.inference import inference_similarity
from protogcn.models import Head, ProtoGCN, Recognizer
from protogcn.tools.run_ensemble import calibrate, fuse


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


ALLOWED_MISSING_CHECKPOINT_KEYS = {
    "cls_head.W",
    "cls_head._current_epoch",
}


def checkpoint_num_classes(state_dict):
    """Infer and cross-check the class count stored in a checkpoint head."""
    dimensions = []
    for key, value in state_dict.items():
        if key.endswith("cls_head.fc_cls.weight") or key.endswith("cls_head.fc_cls.bias"):
            dimensions.append(int(value.shape[0]))
        elif key.endswith("cls_head.csc_loss.avg_f"):
            dimensions.append(int(value.shape[1]))

    if not dimensions:
        raise ValueError("Checkpoint does not contain recognizable class-head tensors")
    if len(set(dimensions)) != 1:
        raise ValueError(f"Checkpoint class-head tensors disagree: {dimensions}")
    return dimensions[0]


def init_demo_recognizer(config_path, checkpoint_path, device):
    """Build a recognizer while allowing only legacy non-training buffers to be absent."""
    config = OmegaConf.load(config_path)
    model_cfg = config.model
    backbone_cfg = {
        key: value
        for key, value in OmegaConf.to_container(model_cfg.backbone, resolve=True).items()
        if key != "type"
    }
    head_cfg = {
        key: value
        for key, value in OmegaConf.to_container(model_cfg.cls_head, resolve=True).items()
        if key != "type"
    }
    train_cfg = OmegaConf.to_container(model_cfg.train_cfg) if "train_cfg" in model_cfg else None
    test_cfg = OmegaConf.to_container(model_cfg.test_cfg) if "test_cfg" in model_cfg else None
    model = Recognizer(
        backbone=ProtoGCN(**backbone_cfg),
        cls_head=Head(**head_cfg),
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    stored_num_classes = checkpoint_num_classes(state_dict)
    expected_num_classes = int(head_cfg["num_classes"])
    if stored_num_classes != expected_num_classes:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has {stored_num_classes} classes, but "
            f"config {config_path} defines {expected_num_classes}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unsupported_missing = set(missing) - ALLOWED_MISSING_CHECKPOINT_KEYS
    if unsupported_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with {config_path}: "
            f"missing={sorted(unsupported_missing)}, unexpected={sorted(unexpected)}"
        )

    model.cfg = config
    model.to(device)
    model.eval()
    return torch.compile(model, mode="default")


def calibrate_and_fuse(probabilities, weights, temperatures):
    """Apply run_ensemble temperature scaling and weighted probability fusion."""
    if len(probabilities) != len(temperatures):
        raise ValueError("Each probability array must have one temperature")

    calibrated = []
    for probs, temperature in zip(probabilities, temperatures):
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        calibrated.append(calibrate(probs, temperature))
    return fuse(calibrated, weights)


class CalibratedEnsembleRecognizer:
    """Demo ensemble using the same calibration and fusion as run_ensemble.py."""

    def __init__(self, streams, device):
        if not streams:
            raise ValueError("At least one ensemble stream is required")

        self.device = device
        self.models = []
        self.pipelines = []
        self.weights = [stream["weight"] for stream in streams]
        self.temperatures = [stream["temperature"] for stream in streams]

        for stream in streams:
            model = init_demo_recognizer(
                stream["config"],
                stream["checkpoint"],
                device,
            )
            self.models.append(model)
            pipeline_src = model.cfg.data.get("test", model.cfg.data.get("val"))
            pipeline_cfg = OmegaConf.to_container(pipeline_src.pipeline)
            self.pipelines.append(Compose(pipeline_cfg))

    def _score_one_stream(self, model, pipeline, keypoint):
        data = {
            "keypoint": keypoint,
            "total_frames": keypoint.shape[1],
            "label": -1,
            "start_index": 0,
            "modality": "Pose",
            "test_mode": True,
        }
        data = pipeline(data)
        processed = data["keypoint"]
        if not isinstance(processed, torch.Tensor):
            processed = torch.FloatTensor(processed)
        processed = processed.to(self.device)

        probabilities, _ = inference_similarity(model, processed)
        return probabilities.mean(0).cpu().numpy()

    def predict(self, keypoint, top_k=5):
        if not isinstance(keypoint, np.ndarray) or keypoint.ndim != 4:
            raise ValueError("keypoint must be a numpy array with shape (M, T, V, C)")

        probabilities = [
            self._score_one_stream(model, pipeline, keypoint)
            for model, pipeline in zip(self.models, self.pipelines)
        ]
        fused = calibrate_and_fuse(
            probabilities,
            self.weights,
            self.temperatures,
        )[0]
        topk_indices = np.argsort(fused)[::-1][:top_k]
        return topk_indices, fused[topk_indices]


class DemoInference:
    """Calibrated ensemble inference for real-time exercise recognition.

    Maintains a rolling frame buffer and re-infers using the last CLIP_LEN
    frames on every call to predict_rolling_window().
    """

    def __init__(self, streams, device=None):
        """
        Args:
            streams: list of dict, each with 'config' (yaml path) and
                'checkpoint' (pth path).
            device: torch device string. Auto-detected if None.
        """
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.ensemble = CalibratedEnsembleRecognizer(streams, device=device)
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
        and runs every configured stream. Temperature-calibrated, weighted
        probabilities determine the predicted class.

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

        # The calibrated ensemble expects (M, T, V, C)
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
