import numpy as np
from scipy.interpolate import interp1d
from fastdtw import fastdtw
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR


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



def masked_dtw_distance(seq1, seq2):
    assert seq1.shape == seq2.shape, "Shape mismatch between sequences"
    T, K, D = seq1.shape
    s1, s2 = [], []
    for t in range(T):
        p1, p2 = [], []
        for k in range(K):
            # 기존에는 -1이면 skip했는데, 이제는 0으로 채운다.
            if (seq1[t, k, 0] == -1 or seq1[t, k, 1] == -1):
                p1.extend([0, 0])
            else:
                p1.extend(seq1[t, k])

            if (seq2[t, k, 0] == -1 or seq2[t, k, 1] == -1):
                p2.extend([0, 0])
            else:
                p2.extend(seq2[t, k])

        s1.append(p1)
        s2.append(p2)

    distance, _ = fastdtw(s1, s2)
    return distance



def sliding_window_dtw(seq, baseline, win_size=30, stride=5):
    distances = []
    for start in range(0, len(seq) - win_size + 1, stride):
        window = seq[start:start + win_size]
        norm_win = normalize_sequence(window, win_size)
        norm_base = normalize_sequence(baseline, win_size)
        dist = masked_dtw_distance(norm_win, norm_base)
        distances.append(dist)
    return distances


def load_all_baselines(baseline_dir):
    """
    baseline_dir 내에 있는 모든 .npz 파일을 불러와 dict로 반환
    """
    baseline_dict = {}
    for fname in os.listdir(baseline_dir):
        if fname.endswith('.npz'):
            key = fname.replace('baseline_', '').replace('.npz', '')
            path = os.path.join(baseline_dir, fname)
            baseline_dict[key] = np.load(path)['keypoints']
    return baseline_dict


def compute_dtw_with_baseline(user_seq, exercise_name, win_size=30, baseline_dir=os.path.join(DATA_DIR, "baseline_means")):
    """
    user_seq: (T, 18, 2)
    exercise_name: 예: 'barbell biceps curl'
    baseline_dir: baseline npz가 있는 디렉토리
    """
    fname = f"baseline_{exercise_name}.npz"
    path = os.path.join(baseline_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")

    baseline = np.load(path)['keypoints']
    if len(baseline) < win_size:
        raise ValueError("Baseline sequence too short")

    user_norm = normalize_sequence(user_seq, win_size)
    base_norm = normalize_sequence(baseline[:win_size], win_size)  # 첫 구간만 비교

    score = masked_dtw_distance(user_norm, base_norm)
    return score


def find_best_matching_baseline(user_seq, baseline_dict, win_size=30):
    min_score = float('inf')
    best_exercise = None

    user_norm = normalize_sequence(user_seq, win_size)

    for exercise_name, baseline_seq in baseline_dict.items():
        if len(baseline_seq) < win_size:
            continue  # baseline이 너무 짧으면 스킵

        for start in range(0, len(baseline_seq) - win_size + 1):
            baseline_window = baseline_seq[start:start + win_size]
            base_norm = normalize_sequence(baseline_window, win_size)
            score = masked_dtw_distance(user_norm, base_norm)

            if score < min_score:
                min_score = score
                best_exercise = exercise_name

    return best_exercise, min_score


def find_best_matching_sequence(user_seq, win_size=30, sequence_dir=os.path.join(DATA_DIR, "sequences")):
    """
    user_seq: (T, 18, 2)
    sequence_dir: 전체 추출된 시퀀스가 저장된 디렉토리
    return: best_file_name, best_score
    """
    min_score = float('inf')
    best_file = None
    user_norm = normalize_sequence(user_seq, win_size)

    for fname in os.listdir(sequence_dir):
        if not fname.endswith('.npz'):
            continue
        seq_data = np.load(os.path.join(sequence_dir, fname))['keypoints']

        if len(seq_data) < win_size:
            continue

        for start in range(0, len(seq_data) - win_size + 1):
            baseline_win = seq_data[start:start + win_size]
            base_norm = normalize_sequence(baseline_win, win_size)
            score = masked_dtw_distance(user_norm, base_norm)

            if score < min_score:
                min_score = score
                best_file = fname

    return best_file, min_score


def analyze_category(user_seq, category, win_size=30, sequence_dir=os.path.join(DATA_DIR, "sequences")):
    scores = []
    user_norm = normalize_sequence(user_seq, win_size)
    for fname in os.listdir(sequence_dir):
        if not fname.endswith('.npz') or category not in fname:
            continue
        file_path = os.path.join(sequence_dir, fname)
        seq = np.load(file_path)['keypoints']
        if len(seq) < win_size:
            continue
        for start in range(0, len(seq) - win_size + 1):
            baseline_win = normalize_sequence(seq[start:start+win_size], win_size)
            score = masked_dtw_distance(user_norm, baseline_win)
            scores.append(score)
    if scores:
        return min(scores), np.mean(scores)
    else:
        return None, None


def normalize_score(score, scale=80000):
    return float(np.clip(np.exp(-score / scale) * 100, 0, 100))
