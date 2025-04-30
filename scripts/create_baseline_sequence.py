import os
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.interpolate import interp1d

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR



def load_sequences(sequences_dir, exercise_name):
    sequences = []
    for fname in os.listdir(sequences_dir):
        if exercise_name.lower() in fname.lower():
            data = np.load(os.path.join(sequences_dir, fname))
            sequences.append(data['keypoints']) # (T, 18, 2)
    return sequences


def normalize_sequence(sequence, target_length=30):
    # (T, K, 2) → (target_len, K, 2)
    T, K, D = sequence.shape
    resampled = np.zeros((target_length, K, D), dtype=np.float32)
    for k in range(K):
        for d in range(D):
            valid = sequence[:, k, d] != -1
            if valid.sum() < 2: # 유효 좌표가 2개 미만이면 보간 불가
                resampled[:, k, d] = -1
                continue
            # 프레임 인덱스를 [0, 1] 구간으로 정규화해서 선형 보간
            f = interp1d(np.linspace(0, 1, T)[valid], sequence[valid, k, d], kind='linear', fill_value='extrapolate')
            resampled[:, k, d] = f(np.linspace(0, 1, target_length))
    return resampled


def create_mean_baseline(sequences, target_len=40):
    aligned = [normalize_sequence(seq, target_len) for seq in sequences]
    stacked = np.stack(aligned)  # (N, T, 18, 2)

    mask = (stacked != -1)
    masked_sum = np.where(mask, stacked, 0).sum(axis=0)  # (T, 18, 2)
    count = mask.sum(axis=0) + 1e-5  # avoid division by zero

    return masked_sum / count



def dtw_distance(a, b):
    distance, _ = fastdtw(a.reshape(len(a), -1), b.reshape(len(b), -1))
    return distance


def select_best_baseline(sequences, target_length=30):
    aligned = [normalize_sequence(seq, target_length) for seq in sequences]
    distances = []
    for i, base in enumerate(aligned):
        total = sum(dtw_distance(base, other) for j, other in enumerate(aligned) if i != j)
        distances.append(total)
    best_idx = np.argmin(distances)
    return aligned[best_idx]


def main(type='mean'):
    sequences_dir = os.path.join(DATA_DIR, 'sequences')
    output_dir = os.path.join(DATA_DIR, 'baseline_means')
    os.makedirs(output_dir, exist_ok=True)
    meta_csv = pd.read_csv(os.path.join(DATA_DIR, 'filter_meta.csv'))
    exercise_names = meta_csv['exercise'].unique().tolist()
    for exercise_name in exercise_names:
        print(f"Processing {exercise_name}...")
        sequences = load_sequences(sequences_dir, exercise_name)
        if not sequences:
            print(f"No sequences found for {exercise_name}.")
            continue
        if type == 'mean':
            baseline_sequence = create_mean_baseline(sequences)
        else:
            baseline_sequence = select_best_baseline(sequences)
        np.savez_compressed(os.path.join(output_dir, f'baseline_{exercise_name}.npz'), keypoints=baseline_sequence)
        print(f"Finished processing {exercise_name}.")


if __name__ == '__main__':
    main()