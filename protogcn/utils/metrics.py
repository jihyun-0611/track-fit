import numpy as np

def intra_class_similarity(features, labels):
    """Compute intra-class cosine similarity mean/std per class.

    Args:
        features (np.ndarray): Feature vectors, shape (N, D).
        labels (np.ndarray): Class labels, shape (N,).

    Returns:
        dict: {
            'per_class': {
                class_id: {'mean': float, 'std': float, 'n_pairs': int}
            },
            'mean' (float): Macro average of the mean by class
            'std' (float): Macro std of the mean by class
        }
    """
    if isinstance(features, list):
        features = np.array(features)
    if isinstance(labels, list):
        labels = np.array(labels)

    # l2
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-12)

    classes = np.unique(labels)
    per_class = {}
    class_means = []

    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue

        feats_c = features[idx] # (n_c, D)
        sim_mat = feats_c @ feats_c.T  # (n_c, n_c) cosine similarity

        # upper triangel -> unique pairs
        triu_idx = np.triu_indices(len(idx), k=1)
        sims = sim_mat[triu_idx] # (n_pairs,)

        per_class[int(c)] = {
            'mean': float(sims.mean()),
            'std': float(sims.std()),
            'n_pairs': len(sims), 
        }
        class_means.append(sims.mean())

    if class_means:
        macro_mean = float(np.mean(class_means))
        macro_std = float(np.std(class_means))
    else:
        macro_mean = float('nan')
        macro_std = float('nan')

    return {
        'per_class': per_class,
        'mean': macro_mean,
        'std': macro_std
    }


def expected_calibration_error(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE).
    to check if the similarity is calibrated.

    ECE = Σ_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args: 
        probs (np.ndarray): softmax probabilities, shape (N, num_classes).
        labels (np.ndarray): ground truth class labels, shape (N,).
        n_bins (int): number of equal-width confidence bins. Default: 15.
    
    Returns:
        dict: {
            'ece' (float): expected calibration error.
            'mce' (float): maximum calibration error (worst bin).
            'bins' (list[dict]): per-bin stats for reliability diagram.
                [{'conf_mean': float, 'acc':float, 'count': int, 'gap': float,...}]
        }
    """
    if isinstance(probs, list):
        probs = np.array(probs)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    confidence = probs.max(axis=1) # (N,) 최대 softmax 확률
    pred = probs.argmax(axis=1) # (N,) 예측 클래스
    correct = (pred==labels).astype(float) # (N,) 정답 여부
     
    N = len(labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # 마지막 bin은 상한 포함
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        count = mask.sum()

        if count == 0:
            bins.append({'conf_mean': (lo+hi) / 2, 'acc': 0.0, 'count': 0, 'gap': 0.0})
            continue

        conf_mean = confidence[mask].mean()
        acc = correct[mask].mean()
        gap = abs(acc - conf_mean)

        ece += (count / N) * gap
        mce = max(mce, gap)
        bins.append({'conf_mean': float(conf_mean), 'acc': float(acc), 
                     'count': int(count), 'gap': float(gap)})
        
    return {'ece': float(ece), 'mce': float(mce), 'bins': bins}

