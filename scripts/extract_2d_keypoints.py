import math
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms

from tqdm import tqdm
import json
import os
import sys

from config import BASE_DIR, WEIGHT_PATH, DATA_DIR

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses


# [x] VideoReader 클래스
class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass
    
    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f'{self.file_name} cannot be opened')
        return self
    
    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img
    

def normalize(img, img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    img = np.array(img, dtype=np.float32)
    return (img - img_mean) * img_scale

def pad_width(img, stride, min_dims, pad_value=(0, 0, 0)):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


# TODO 프레임 단위 추론
# [x] infer_fast() : heatmap/paf 추출 (upsample 포함)
def infer_frame(net, device, frame, net_input_height, stride, upsample_ratio):
    h, w, _ = frame.shape
    scale = net_input_height / h

    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = normalize(img)

    min_dims = [net_input_height, max(img.shape[1], net_input_height)]
    img, pad = pad_width(img, stride, min_dims)

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img = img.to(device)

    # inference
    stages_output = net(img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def pose_to_dict(pose):
    return {
        "id": pose.id,
        "keypoints": [
            {"x": float(x), "y": float(y)} for (x, y) in pose.keypoints
        ],
        "bbox": [int(pose.bbox[0]), int(pose.bbox[1]), int(pose.bbox[2]), int(pose.bbox[3])],
        "confidence": float(pose.confidence)
    }


# TODO 영상 프로세스
# [x] extract_keypoints() + group_keypoints() -> 후처리 알고리즘 
# [x] 좌표 복원 : heatmap -> 원본이미지 좌표계로 보정 
# [x] Pose 객체 -> 딕셔너리로 변환
# [x] frame_index 별 keypoints 리스트 구성 
def process_video(net, device, frame_provider, net_input_height=256, smooth=1):
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts # 18개 
    frame_keypoints_list = []
    previous_poses = []

    frame_idx = 0
    for frame in frame_provider:
        origin_img = frame.copy()
        # inference
        heatmaps, pafs, scale, pad = infer_frame(net, device, frame, 
                                                 net_input_height, 
                                                 stride, upsample_ratio)

        # post precessing
        # keypoints 후보 추출
        total_keypoints_num = 0
        all_keypoints_by_type = [] # 
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], 
                all_keypoints_by_type, 
                total_keypoints_num)

        # 각 사람 단위로 연결
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        # all_keypoints.shape = (total_num_keypoints, 4) : (x, y, confidence, global id)
        # 좌표 복원
        for kpt_id in range(all_keypoints.shape[0]): 
            # x 좌표
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            # y 좌표
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        # pose 객체 생성
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                # pose_entries[n][kpt_id]는 해당 사람의 k번째 관절의 global keypoint index
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        # track pose -> id 유지
        track_poses(previous_poses, current_poses, smooth=smooth)
        frame_data = {
            "frame_index": frame_idx, 
            "poses": [pose_to_dict(pose) for pose in current_poses]
        }
        frame_keypoints_list.append(frame_data)
        previous_poses = current_poses
        frame_idx += 1
    # [x] 영상 단위로 keypoint 저장
    return frame_keypoints_list


# TODO 파일로 저장
# [x] 최종 .json 파일로 dump
def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "filter_meta.csv"))

    video_paths = []
    for idx, row in df.iterrows():
        path = os.path.join('sample_videos', row['exercise'], row['file_name'])
        video_paths.append(path)
    print(f"총 {len(video_paths)}개의 비디오 데이터")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    load_state(net, checkpoint)
    net = net.to(device).eval()

    output_path = os.path.join(DATA_DIR, "keypoints")
    os.makedirs(output_path, exist_ok=True)
    for video_path in tqdm(video_paths):
        start_time = time.time()

        file_path = os.path.join(DATA_DIR, video_path)
        frame_provider = VideoReader(file_path)

        result = process_video(net, device, frame_provider)
        filename = os.path.splitext(os.path.basename(video_path))[0]
        with open(os.path.join(output_path, f"{filename}.json"), 'w') as f:
            json.dump(result, f)
        
        end_time = time.time()
        print(f"time: {end_time-start_time}\n")
        print(f"{os.path.join(output_path, f"{filename}.json")} 추출 완료")


if __name__ == '__main__':
    main()
