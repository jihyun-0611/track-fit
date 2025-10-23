import numpy as np


MEDIAPIPE_TO_COCO = {
    0: 0, # nose
    2: 1, # left_eye
    5: 2, # right_eye
    7: 3, # left_ear
    8: 4, # right_ear
    11: 5, # left_shoulder
    12: 6, # right_shoulder
    13: 7, # left_elbow
    14: 8, # right_elbow
    15: 9, # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
    # COCO_NEW
    19: 17, # left_big_toe
    20: 18, # left_small_toe
    31: 19, # right_big_toe
}

def mediapipe_to_coco(mp_landmarks, width, height):
    """MediaPipe format에서 COCO format으로 변환 (정규화된 좌표 유지)"""
    keypoints = np.zeros((20, 3), dtype=np.float32)

    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
        landmark = mp_landmarks.landmark[mp_idx]
        keypoints[coco_idx] = [
            landmark.x,  # 정규화된 좌표 (0-1) 유지
            landmark.y,  # 정규화된 좌표 (0-1) 유지
            landmark.visibility
        ]

    return keypoints

