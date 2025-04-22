import os
import json

import numpy as np
from tqdm import tqdm
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, CHECKPOINT, DATA_DIR
sys.path.append(os.path.join(BASE_DIR, 'external/lightweight-human-pose-estimation.pytorch'))
from scripts.train_pose3d_baseline import LinearModel


H36M_USED_JOINTS = [
    0,   # Hip
    1, 2, 3,      # Right leg
    6, 7, 8,      # Left leg
    12, 13,       # Spine, Thorax
    15,           # Head
    17, 18, 19,   # Left arm
    25, 26, 27    # Right arm
]
OPENPOSE_TO_H36M = {
    0: 15,  # nose → Head
    1: 13,  # neck → Thorax
    2: 25, 3: 26, 4: 27,  # r_sho, r_elb, r_wri
    5: 17, 6: 18, 7: 19,  # l_sho, l_elb, l_wri
    8: 1, 9: 2, 10: 3,    # r_hip, r_knee, r_ank
    11: 6, 12: 7, 13: 8   # l_hip, l_knee, l_ank
}


def load_keypoints_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    

def convert_openpose18_to_h36(pose_kpts):
    """
    OpenPose로 예측한 18-keypoint format을 학습한 모델에 사용한 Human3.6M 17-joint format으로 변환

    Parameters:
        pose_18x2 (np.ndarray): shape (18, 2) keypoints from OpenPose (x, y)
    
    Returns:
        pose_17x2 (np.ndarray): shape (17, 2) keypoints in Human3.6M training format, with missing values as -1
    """
    
    pose_32 = np.full((32, 2), -1.0, dtype=np.float32)

    for open_idx, h36m_idx in OPENPOSE_TO_H36M.items():
        if pose_kpts[open_idx, 0] != -1:
            pose_32[h36m_idx] = pose_kpts[open_idx]

    pose_16 = pose_32[H36M_USED_JOINTS]

    return pose_16


def load_model(device):
    net = LinearModel(input_size=32, output_size=96).to(device)
    checkpoint = torch.load(os.path.join(CHECKPOINT, "Best model_39"), map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    return net


def inference(kpts, model, device):
    inputs = torch.tensor(kpts, dtype=torch.float32).view(1, -1).to(device)
    
    with torch.no_grad():
        outputs = model(inputs).view(-1, 32, 3).cpu().numpy()
    return outputs


def process_keypoint(data_path): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(device)

    file_list = [f for f in os.listdir(data_path) if f.endswith(".json")]
    for file_name in file_list: # process keypoints of one video
        video_keypoints = load_keypoints_json(os.path.join(data_path, file_name))
        for frame_keypoints in video_keypoints: # process keypoints of one frame
            if len(frame_keypoints['poses']) == 0:
                continue
            for person in frame_keypoints['poses']:
                kpts = [[kpt['x'], kpt['y']] for kpt in person['keypoints']]
                convert_kpts = convert_openpose18_to_h36(np.array(kpts))
                output_3d = inference(convert_kpts, net, device)
                used_3d_output = output_3d[0][H36M_USED_JOINTS]
                person['keypoints_3d'] = [
                    {'x': float(x), 'y': float(y), 'z': float(z)} for x, y, z in used_3d_output
                ]

            # one_pose = frame_keypoints['poses'][0]
            # kpts = [[kpt['x'], kpt['y']]for kpt in one_pose['keypoints']]
            # convert_kpts = convert_openpose18_to_h36(np.array(kpts))
            # output_3d = inference(convert_kpts, net, device)
            # used_3d_output = output_3d[0][H36M_USED_JOINTS]
            # frame_keypoints['poses'][0]['keypoints_3d'] = [
            #     {'x': float(x), 'y': float(y), 'z': float(z)} for x, y, z in used_3d_output
            # ]
        # JSON 저장
        save_path = os.path.join(data_path, file_name.replace('.json', '_with_3d.json'))
        with open(save_path, 'w') as f:
            json.dump(video_keypoints, f, indent=2)


if __name__ == '__main__':
    data_path = os.path.join(DATA_DIR, 'keypoints')
    process_keypoint(data_path)
