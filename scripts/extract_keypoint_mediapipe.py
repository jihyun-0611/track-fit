import time
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm
import json
import os
import argparse
import pickle
from pathlib import Path


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


NUM_COCO_KEYPOINTS = 17

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

}


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
    

def mediapipe_to_coco(mp_landmarks, img_width, img_height):
    """
    MediaPipe 33 keypoints를 COCO 17 keypoints로 변환
    Returns:
        np.array: shape (17, 3) - [x, y, confidence]
    """

    coco_kps = np.zeros((NUM_COCO_KEYPOINTS, 3), dtype=np.float32)

    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
        landmark = mp_landmarks.landmark[mp_idx]
        coco_kps[coco_idx] = [
            landmark.x * img_width,
            landmark.y * img_height,
            landmark.visibility # confidence
        ]

    return coco_kps


def process_video(frame_provider, min_detection_confidence=0.5, 
                  min_tracking_confidence=0.5):
    """
    MediaPipe를 통해 비디오에서 Keypoints 추출
    Returns:
        list: 프레임별 keypoints 정보
    """

    frame_keypoints_list = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as pose:
        
        frame_idx = 0

        for frame in frame_provider:
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(frame_rgb)

            frame_data = {
                "frame_idx": frame_idx,
                "poses": []
            }

            if results.pose_landmarks:
                coco_keypoints = mediapipe_to_coco(
                    results.pose_landmarks, w, h
                )

                valid_keypoints = coco_keypoints[coco_keypoints[:, 2] >= 0.3]

                if len(valid_keypoints) > 0:
                    valid_coords = coco_keypoints[coco_keypoints[:, 2] >= 0.3][:, :2]
                    if len(valid_coords) > 0:
                        x_min = np.min(valid_coords[:, 0])
                        y_min = np.min(valid_coords[:, 1])
                        x_max = np.max(valid_coords[:, 0])
                        y_max = np.max(valid_coords[:, 1])
                        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                    else:
                        bbox = [0, 0, w, h]

                    avg_confidence = np.mean(coco_keypoints[:, 2])

                    pose_data = {
                        "id": 0, 
                        "keypoints": [
                            {
                                "x": float(kp[0]),
                                "y": float(kp[1]),
                                "confidence": float(kp[2])
                            }
                            for kp in coco_keypoints
                        ],
                        "bbox": bbox,
                        "confidence": float(avg_confidence)
                    }

                    frame_data["poses"].append(pose_data)

            frame_keypoints_list.append(frame_data)
            frame_idx += 1

    return frame_keypoints_list


def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_to_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description='Extract keypoints from exercise videos using MediaPipe')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get('DATA_DIR'),
        help='Data directory path (default: $DATA_DIR from .env)'
    )
    parser.add_argument(
        '--sub-dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Metadata CSV path (default: <data-dir>/meta.csv)'
    )
    parser.add_argument(
        '--min-detection-confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence for MediaPipe (default: 0.5)'
    )
    parser.add_argument(
        '--min-tracking-confidence',
        type=float,
        default=0.5,
        help='Minimum tracking confidence for MediaPipe (default: 0.5)'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Also save extracted keypoints as JSON files (default: disabled)'
    )
    args = parser.parse_args()

    if not args.data_dir:
        print("Error: DATA_DIR not set. Please set DATA_DIR in .env file or use --data-dir argument.")
        return

    data_dir = Path(args.data_dir)
    sub_dir = Path(args.sub_dir)
    csv_path = (data_dir / Path(args.csv_path)) if args.csv_path else data_dir / "meta.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(csv_path)

    video_paths = []
    for idx, row in df.iterrows():
        path = os.path.join(sub_dir, row['exercise'], row['file_name'])
        video_paths.append({
            'path':path,
            'exercise': row['exercise'],
            'filename': row['file_name']
        })

    print(f"Data directory: {data_dir}")
    print(f"CSV path: {csv_path}")
    print(f"총 {len(video_paths)}개의 비디오 데이터")


    pickle_output_dir = data_dir / "keypoints" / "pickle"

    if args.save_json:
        json_output_dir = data_dir / "keypoints" / "json"
        json_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        json_output_dir = None
    pickle_output_dir.mkdir(parents=True, exist_ok=True)

    total_videos = len(video_paths)
    success_count = 0
    fail_count = 0
    failed_videos = []

    for video_info in tqdm(video_paths, desc="Processing videos"):
        start_time = time.time()

        video_path = video_info['path']
        file_path = data_dir / video_path

        if not file_path.exists():
            print(f"\nWarning: {file_path} 파일을 찾을 수 없어 건너 뜁니다.")
            fail_count +=1
            failed_videos.append(video_path)
            continue

        try:
            frame_provider = VideoReader(str(file_path))

            result = process_video(
                frame_provider,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence
            )

            filename = os.path.splitext(video_info['filename'])[0]
            exercise = video_info['exercise']

            pickle_exercise_dir = pickle_output_dir / exercise
            pickle_exercise_dir.mkdir(parents=True, exist_ok=True)

            if args.save_json:
                json_exercise_dir = json_output_dir / exercise
                json_exercise_dir.mkdir(parents=True, exist_ok=True)
                json_output_path = json_exercise_dir / f"{filename}.json"
                save_to_json(result, str(json_output_path))

            pickle_output_path = pickle_exercise_dir / f"{filename}.pkl"
            save_to_pickle(result, str(pickle_output_path))

            end_time = time.time()
            elapsed_time = end_time - start_time

            total_frames = len(result)
            frames_with_pose = sum(1 for frame in result if len(frame['poses'])>0)
            detection_rate = (frames_with_pose/total_frames*100) if total_frames > 0 else 0

            success_count += 1

            tqdm.write(
                f"\n {filename} | "
                f"시간: {elapsed_time:.2f}초 | "
                f"프레임: {total_frames} | "
                f"감지율: {detection_rate:.1f}%"
            )

        except Exception as e:
            fail_count += 1
            failed_videos.append(video_path)
            tqdm.write(f"\n Error processing {video_path}: {str(e)}")
            continue

    
    print("\n" + "="*60)
    print(f"총 비디오 수: {total_videos}")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")
    
    if failed_videos:
        print("\n실패한 비디오 목록")
        for video in failed_videos:
            print(f"  - {video}")

    print(f"\noutput directory:")
    if args.save_json:
        print(f"    JSON: {json_output_dir}")
    print(f"    Pickle: {pickle_output_dir}")
    print("="*60)

if __name__=='__main__':
    main()

