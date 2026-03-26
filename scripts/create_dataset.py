import json
import pickle
import numpy as np
import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    total_frames = max(d['frame_idx'] for d in data) + 1
    keypoint = np.zeros((1, total_frames, 20, 2), dtype=np.float32)
    keypoint_score = np.zeros((1, total_frames, 20), dtype=np.float32)

    for frame_data in data:
        fi = frame_data['frame_idx']
        if frame_data['poses']:
            for j, kp in enumerate(frame_data['poses'][0]['keypoints'][:20]):
                keypoint[0, fi, j, 0] = kp['x']
                keypoint[0, fi, j, 1] = kp['y']
                keypoint_score[0, fi, j] = kp['confidence']
    return keypoint, keypoint_score


def load_from_json(json_path):
    """
    Returns:
        keypoint: shape (M, T, V, C) - M=1, T=프레임수, V=20, C=2
        keypoint_score: shape (M, T, V)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_frames = max(d['frame_idx'] for d in data) + 1
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


def make_clips(keypoint, keypoint_score, base_name, label,
               clip_len=100, stride=50, min_frames=30):
    """
    - T < min_frames : 제외
    - T < clip_len : 전체 영상을 1개 클립으로
    - T >= clip_len : 슬라이딩 윈도우
    """
    T = keypoint.shape[1]
    if T < min_frames:
        return []
    
    clips = []
    if T < clip_len:
        clips.append({
            'frame_dir': base_name,
            'total_frames': T,
            'label': label,
            'keypoint': keypoint,
            'keypoint_score': keypoint_score
        })
    else:
        for i, start in enumerate(range(0, T-min_frames+1, stride)):
            end = min(start + clip_len, T)
            if end - start < min_frames:
                break
            clips.append({
                'frame_dir': f'{base_name}_clip{i}',
                'total_frames': end - start,
                'label': label,
                'keypoint': keypoint[:, start:end].copy(),
                'keypoint_score': keypoint_score[:, start:end].copy()
            })
    return clips


def create_dataset(kpt_dir, csv_path, output_path, 
                   val_ratio=0.1, test_ratio=0.1, random_seed=42,
                   clip_len=100, stride=50, min_frames=30,
                   exclude_labels=None):
    
    df = pd.read_csv(csv_path)

    if exclude_labels:
        df = df[~df['exercise'].isin(exclude_labels)]

    exercise_types = sorted(df['exercise'].unique())
    label_mapping = {ex: idx for idx, ex in enumerate(exercise_types)}

    print(f"클래스 수 : {len(exercise_types)}")
    print(f"클래스 매핑: {label_mapping}")

    # video 단위로 train/val 분리 
    from collections import defaultdict
    video_by_class = defaultdict(list)
    for _, row in df.iterrows():
        video_by_class[row['exercise']].append(row['file_name'])

    train_files, val_files, test_files = set(), set(), set()
    for ex, files in video_by_class.items():
        tr_va, te = train_test_split(files, test_size=test_ratio,
                                     random_state=random_seed)
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        tr, va = train_test_split(tr_va, test_size=val_ratio_adjusted, 
                                  random_state=random_seed)
        train_files.update(tr)
        val_files.update(va)
        test_files.update(te)

    # 클립 생성
    annotations, train_ids, val_ids, test_ids = [], [], [], []
    skipped = []

    for _, row in df.iterrows():
        exercise = row['exercise']
        file_name = row['file_name']
        base_name = os.path.splitext(file_name)[0]

        pkl_path = os.path.join(kpt_dir, 'pickle', exercise, f'{base_name}.pkl')
        json_path = os.path.join(kpt_dir, 'json', exercise, f'{base_name}.json')

        if os.path.exists(pkl_path):
            keypoint, keypoint_score = load_from_pickle(pkl_path)
        elif os.path.exists(json_path):
            keypoint, keypoint_score = load_from_json(json_path)
        else:
            skipped.append(base_name)
            continue

        clips = make_clips(keypoint, keypoint_score, base_name,
                           label_mapping[exercise], clip_len=clip_len,
                           stride=stride, min_frames=min_frames)
        
        for clip in clips:
            annotations.append(clip)
            if file_name in train_files:
                train_ids.append(clip['frame_dir'])
            elif file_name in val_files:
                val_ids.append(clip['frame_dir'])
            else:
                test_ids.append(clip['frame_dir'])
    
    dataset = {
        'split': {'train': train_ids, 'val': val_ids, 'test': test_ids},
        'annotations': annotations
    }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n데이터셋 생성 완료 : {output_path}")
    print(f"  Train: {len(train_ids)}개")
    print(f"  Val: {len(val_ids)}개")
    print(f"  Test: {len(test_ids)}개")

    print(f"\n클래스 분포: ")
    frame_dir_to_split = {}
    for fid in train_ids: frame_dir_to_split[fid] = 'train'
    for fid in val_ids: frame_dir_to_split[fid] = 'val'
    for fid in test_ids: frame_dir_to_split[fid] = 'test'

    for exercise, label in label_mapping.items():
        class_annos = [a for a in annotations if a['label'] == label]
        tr = sum(1 for a in class_annos if frame_dir_to_split.get(a['frame_dir']) == 'train')
        va = sum(1 for a in class_annos if frame_dir_to_split.get(a['frame_dir']) == 'val')
        te = sum(1 for a in class_annos if frame_dir_to_split.get(a['frame_dir']) == 'test')

        print(f"  {exercise} (label={label}): {len(class_annos)}개 (train={tr}, val={va}, test={te})")

    label_mapping_path = str(Path(output_path).with_suffix('')) + '_label_mapping.json'
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n라벨 매핑 정보 저장: {label_mapping_path}")

    if skipped:
        print(f"\nNum of skipped files ({len(skipped)}): {skipped}")


def verify_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    print("\n" + "="*60)
    print(f"데이터셋 정보: {pkl_path}")
    print("="*60)

    print(f"Train samples: {len(dataset['split']['train'])}")
    print(f"Val samples: {len(dataset['split']['val'])}")
    print(f"Test samples: {len(dataset['split']['test'])}")

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
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description='Create exercise dataset from keypoints')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get('DATA_DIR'),
        help='Data directory path (default: $DATA_DIR from .env)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='exercise_dataset.pkl'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='test split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--exclude-labels',
        type=str,
        nargs='+',
        default=None,
        help='List of labels to exclude (default: None)'
    )

    args = parser.parse_args()

    if not args.data_dir:
        print("Error: DATA_DIR not set. Please set DATA_DIR in .env file or use --data-dir argument.")
        return

    data_dir = Path(args.data_dir)
    kpt_dir = data_dir / "keypoints"
    csv_path = data_dir / "meta.csv"
    output_path = data_dir / args.output_path
    if not kpt_dir.exists():
        print(f"Error: {kpt_dir} 디렉토리를 찾을 수 없습니다.")
        return
    if not csv_path.exists():
        print(f"Error: {csv_path} 파일을 찾을 수 없습니다.")
        return

    print(f"Data directory: {data_dir}")
    print(f"Keypoints directory: {kpt_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Output path: {output_path}\n")

    create_dataset(str(kpt_dir), str(csv_path), str(output_path),
                   val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                   random_seed=args.random_seed,
                   exclude_labels=args.exclude_labels)
    verify_dataset(str(output_path))
    print("데이터셋 생성 및 검증 완료.")

if __name__ == '__main__':
    main()