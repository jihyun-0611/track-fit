import numpy as np


def confusion_matrix(y_pred, y_real, normalize=None):
    """
    Confusion_matrix
    
    :param y_pred: (list[int] | np.ndarray[int])
    :param y_real: (list[int] | np.ndarray[int])
    :param normalize: (str | None) 
            Options for normalization 
            "true(rows)", "predicted(columns)", "all", None
    """

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")
    
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(f'y_pred dtype must be np.int64, but got {y_pred.dtype}')
    
    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')
    
    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped, # idx = (row * {전체 열의 개수}) + col
        minlength = num_labels **2).reshape(num_labels, num_labels)
    
    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True)
            )
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True)
            )
        elif normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum()
            )
        confusion_mat = np.nan_to_num(confusion_mat)
    return confusion_mat


def top_k_accuracy(scores, labels, topk=(1, )):
    """
    Calculate  top_k_accuracy score
    
    Args:
        scores (np.ndarray): Prediction scores for each class, 
            shape (N, num_classes).
        labels (np.ndarray): Ground truth labels.
            shape (N,)
        topk (tuples): Top-k values to compute accuracy for.

    :return list[float]: top k accuracy score for each k 
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

