import numpy as np


MEDIAPIPE_TO_COCO17 = {
    0: 0,  # nose
    2: 1,  # left_eye
    5: 2,  # right_eye
    7: 3,  # left_ear
    8: 4,  # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}


def mediapipe_to_coco20(mp_landmarks, width, height):
    """Convert MediaPipe landmarks to the COCO20 format used for training."""
    keypoints = np.zeros((20, 3), dtype=np.float32)

    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO17.items():
        landmark = mp_landmarks.landmark[mp_idx]
        keypoints[coco_idx] = [
            landmark.x * width,
            landmark.y * height,
            landmark.visibility,
        ]

    mid_hip = (keypoints[11] + keypoints[12]) / 2.0
    mid_shoulder = (keypoints[5] + keypoints[6]) / 2.0
    spine = (mid_hip + mid_shoulder) / 2.0
    keypoints[17] = mid_hip
    keypoints[18] = spine
    keypoints[19] = mid_shoulder

    return keypoints
