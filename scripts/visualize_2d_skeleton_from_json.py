import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, WEIGHT_PATH, DATA_DIR
sys.path.append(os.path.join(BASE_DIR, 'external/lightweight-human-pose-estimation.pytorch'))
from modules.pose import Pose


def load_keypoints_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    

def dict_to_pose(pose_dict):
    keypoints_array = np.array([
        [kpt["x"], kpt["y"]] for kpt in pose_dict["keypoints"]
    ], dtype=np.float32)

    pose = Pose(keypoints_array, confidence=pose_dict["confidence"])
    if "id" in pose_dict:
        pose.id = pose_dict["id"]

    return pose

def visualize_pose_json_on_video(video_path, json_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    keypoints_data = load_keypoints_json(json_path)

    # 저장
    save = save_path is not None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 프레임 별 처리
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints_data):
            break

        # pose 복원
        frame_data = keypoints_data[frame_idx]
        poses = [dict_to_pose(pose_dict) for pose_dict in frame_data["poses"]]

        for pose in poses:
            pose.draw(frame)

        if save:
            out.write(frame)
        else:
            cv2.imshow("Pose Visualization", frame)
            if cv2.waitKey(1) == 27:
                break
        
        frame_idx += 1

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = os.path.join(DATA_DIR, "sample_videos/lat pulldown/lat pulldown_1.mp4")
    json_path = os.path.join(DATA_DIR, "keypoints/lat pulldown_1.json")
    save_path = os.path.join(DATA_DIR, "visualized_lat pulldown_1.mp4") 
    visualize_pose_json_on_video(video_path, json_path, save_path)

