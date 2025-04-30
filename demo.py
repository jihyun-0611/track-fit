import cv2
import torch
import numpy as np
from collections import deque
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, WEIGHT_PATH, DATA_DIR
sys.path.append(os.path.join(BASE_DIR, 'external/lightweight-human-pose-estimation.pytorch'))
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses

from utils.sequence_utils import normalize_score, load_all_baselines, compute_dtw_with_baseline
from utils.pose_utils import normalize, pad_width, infer_frame


def main(baseline_dir, exercise_name, video_path=None, save_output=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    load_state(net, checkpoint)
    net = net.to(device).eval()


    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # 웹캠 사용
        cap = cv2.VideoCapture(0)
    

    # 비디오 저장 설정
    if save_output:
        output_path = os.path.join(DATA_DIR, 'output_demo.mp4')

        # 원본 비디오의 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # VideoWriter 객체 생성 (코덱: mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    net_input_height = 256
    stride = 8
    upsample_ratio = 4
    win_size = 30

    frame_buffer = deque(maxlen=win_size)
    previous_poses = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = frame.copy()

        # 2d pose 추론
        heatmaps, pafs, scale, pad = infer_frame(net, device, frame, net_input_height, stride, upsample_ratio)

        # keypoints 추출
        num_keypoints = Pose.num_kpts
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        # 좌표 복원
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        # pose 객체 생성
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            confidence = pose_entries[n][18] if len(pose_entries[n]) > 18 else 1.0
            pose = Pose(pose_keypoints, confidence)
            current_poses.append(pose)

        # pose tracking
        track_poses(previous_poses, current_poses, smooth=True)
        previous_poses = current_poses

        # 가장 confidence 높은 사람만 선택
        if current_poses:
            best_pose = max(current_poses, key=lambda p: p.confidence)
            frame_buffer.append(best_pose.keypoints)

        # buffer가 꽉 차면 DTW score 계산
        score = None
        if len(frame_buffer) == win_size:
            seq = np.stack(frame_buffer)
            score = compute_dtw_with_baseline(seq, exercise_name, win_size, baseline_dir)


        # 시각화 
        for pose in current_poses:
            pose.draw(frame)
        if score is not None:
            visual_score = normalize_score(score)
            text = f"{exercise_name} | Score: {visual_score:.1f}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        if save_output:
            out.write(frame)

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if save_output:
        out.release()



if __name__ == "__main__":
    video_path = os.path.join(DATA_DIR, 'sample_videos/barbell biceps curl/barbell biceps curl_62.mp4')
    exercise_name="barbell biceps curl"
    baseline_dir = os.path.join(DATA_DIR, 'baseline_best') # or 'baseline_means'
    main(baseline_dir, exercise_name, video_path, save_output=True)