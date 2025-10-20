import json
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_DIR


def load_keypoints_from_json(json_path):
    """
    Returns:
        keypoint: shape (M, T, V, C) - M=1, T=프레임수, V=17, C=3
        keypoint_score: shape (M, T, V)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_frames = len(data)
    num_person = 1
    num_joints = 20

    keypoint = np.zeros((num_person, total_frames, num_joints, 2), dtype=np.float32)
    keypoint_score = np.zeros((num_person, total_frames, num_joints), dtype=np.float32)

    for frame_data in data:
        frame_idx = frame_data['frame_idx']
        poses = frame_data.get('poses', [])

        if poses:
            pose = poses[0]
            keypoints_list = pose['keypoints']

            for joint_idx, kp in enumerate(keypoints_list):
                if joint_idx >= num_joints:
                    break

                keypoint[0, frame_idx, joint_idx, 0] = kp['x']
                keypoint[0, frame_idx, joint_idx, 1] = kp['y']
                keypoint_score[0,  frame_idx, joint_idx] = kp['confidence']
    
    return keypoint, keypoint_score


def create_dataset(json_dir, csv_path, output_path, 
                   train_ratio=0.8, random_seed=42):
    
    df = pd.read_csv(csv_path)

    exercise_types = sorted(df['exercise'].unique())
    label_mapping = {ex: idx for idx, ex in enumerate(exercise_types)}

    print(f"클래스 수 : {len(exercise_types)}")
    print(f"클래스 매핑: {label_mapping}")

    annotations = []
    skipped_videos = []

    for idx, row in df.iterrows():
        exercise = row['exercise']
        file_name = row['file_name']
        base_name = os.path.splitext(file_name)[0]

        json_path = os.path.join(json_dir, exercise, f"{base_name}.json")

        if not os.path.exists(json_path):
            print(f"Warning: {json_path}파일을 찾을 수 없습니다. ")
            skipped_videos.append(base_name)
            continue

        try:
            keypoint, keypoint_score = load_keypoints_from_json(json_path)

            annotation = {
                'frame_dir': base_name, # 비디오 id
                'total_frames': keypoint.shape[1],
                'label': label_mapping[exercise],
                'keypoint': keypoint,          # (1, T, 17, 3)
                'keypoint_score': keypoint_score # (1, T, 17)
            }

            annotations.append(annotation)

        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            skipped_videos.append(base_name)
            continue

    print(f"총 {len(annotations)}개의 어노테이션 생성")
    if skipped_videos:
        print(f"건너뛴 비디오 수: {len(skipped_videos)}")


    video_ids = [anno['frame_dir'] for anno in annotations]
    labels = [anno['label'] for anno in annotations]

    train_ids, val_ids = train_test_split(
        video_ids,
        test_size=1 - train_ratio,
        random_state=random_seed,
        stratify=labels
    )

    dataset = {
        'split': {
            'train': train_ids,
            'val': val_ids
        },
        'annotations': annotations
    }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n데이터셋 생성 완료 : {output_path}")
    print(f"  Train: {len(train_ids)}개")
    print(f"  Val: {len(val_ids)}개")

    print(f"\n클래스 분포: ")
    for exercise, label in label_mapping.items():
        count = sum(1 for anno in annotations if anno['label'] == label)
        train_count = sum(1 for vid in train_ids
                          if any(anno['frame_dir'] == vid and anno['label'] == label 
                                 for anno in annotations))
        val_count = count - train_count
        print(f"  {exercise} (label={label}): {count}개 (train={train_count}, val={val_count})")

    label_mapping_path = output_path.replace('.pkl', '_label_mapping.json')
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n라벨 매핑 정보 저장: {label_mapping_path}")

    if skipped_videos:
        print(f"\n건너뛴 비디오 목록 (총 {len(skipped_videos)}개):")
        for video in skipped_videos[:10]:  # 최대 10개만 출력
            print(f"  - {video}")
        if len(skipped_videos) > 10:
            print("  ...외 {len(skipped_videos) - 10}개")


def verify_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    print("\n" + "="*60)
    print(f"데이터셋 정보: {pkl_path}")
    print("="*60)

    print(f"Train samples: {len(dataset['split']['train'])}")
    print(f"Val samples: {len(dataset['split']['val'])}")

    print(f"\n Anntations : {len(dataset['annotations'])}")

    
    # 첫번째 샘플 정보 출력
    if dataset['annotations']:
        sample = dataset['annotations'][0]
        print("\n샘플 어노테이션 예시:")
        print(f"  frame_dir: {sample['frame_dir']}")
        print(f"  total_frames: {sample['total_frames']}")
        print(f"  label: {sample['label']}")
        print(f"  keypoint shape: {sample['keypoint'].shape}") # (M, T, V, C)
        print(f"  keypoint_score shape: {sample['keypoint_score'].shape}") # (M, T, V)
    
    print("="*60)


def main():
    json_dir = os.path.join(DATA_DIR, "keypoints_mediapipe", "json")
    csv_path = os.path.join(DATA_DIR, "filter_meta.csv")
    output_path = os.path.join(DATA_DIR, "exercise_dataset.pkl")

    if not os.path.exists(json_dir):
        print(f"Error: {json_dir} 디렉토리를 찾을 수 없습니다.")
        return
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} 파일을 찾을 수 없습니다.")
        return

    create_dataset(json_dir, csv_path, output_path)
    verify_dataset(output_path)
    print("데이터셋 생성 및 검증 완료.")

if __name__ == '__main__':
    main()