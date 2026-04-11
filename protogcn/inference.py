import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
from omegaconf import DictConfig, OmegaConf

from .models import Recognizer, ProtoGCN, Head
from .datasets.pipelines import Compose
from .utils import OutputHook


def init_recognizer(config, checkpoint=None, device='cuda:0'):
    """Initialize a recognizer from config.
    
    Args:
        config (str | dict | DictConfig): Config file path (.yaml) or config dict / DictConfig object.
        checkpoint (str | None): Checkpoint path. If None, no weights are loaded. Default: None.
        device (str): Target device. Default: 'cuda:0'.
    
    Returns:
        nn.Module: The constructed recognizer
    """
    if isinstance(config, str):
        config = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)
    elif not isinstance(config, DictConfig):
        raise TypeError(
            f'config must be a str, dict, or DictConfig, but got {type(config)}'
        )
    
    model_cfg = config['model']
    backbone_cfg = {
        k: v for k, v in OmegaConf.to_container(model_cfg['backbone']).items()
        if k != 'type'
    }
    head_cfg = {
        k: v for k, v in OmegaConf.to_container(model_cfg['cls_head']).items()
        if k!= 'type'
    }
    train_cfg = OmegaConf.to_container(model_cfg['train_cfg']) if 'train_cfg' in model_cfg else None
    test_cfg = OmegaConf.to_container(model_cfg['test_cfg']) if 'train_cfg' in model_cfg else None

    model = Recognizer(
        backbone=ProtoGCN(**backbone_cfg),
        cls_head=Head(**head_cfg),
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model, data, outputs=None, as_tensor=True, top_k=5):
    """Inference a skeleton sequence with the recognizer.
    
    Args:
        model (nn.Module): The loaded recognizer.
        data (dict | np.ndarray): Annotation-style dict (with 'keypoint', 'total_frames', etc.) or 
                                a numpy array of shape (M, T, V, C).
        outputs (list[str] | tuple[str] | str | None): Names of layers whose outputs need to be returned.
                                Default: None.
        as_tensor (bool): Same as that in ``OutputHook``. Default: True.
        top_k (int): Number of top predictions to return. Default: 5.
    
    Returns:
        list[tuple(int, flaot)]: Top-k (class_index, score) pairs.
        dict (optional): Layer outputs if ``outputs`` is specified.
    """
    if isinstance(outputs, str):
        outputs = (outputs,)
    assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device

    if isinstance(data, np.ndarray):
        assert data.ndim == 4, 'ndarray input must have shape (M, T, V, C)'
        data = dict(
            keypoint=data,
            total_frames=data.shape[1],
            label=-1,
            start_index=0,
            modality='Pose',
            test_mode=True,
        )
    elif not isinstance(data, dict):
        raise TypeError(f'data must be a dict or np.ndarray, but got {type(data)}')
    
    # Build test pipeline from config (fall back to val pipeline if test is absent)
    data_cfg = cfg.data
    pipeline_src = data_cfg.get('test', data_cfg.get('val'))
    pipeline_cfg = OmegaConf.to_container(pipeline_src.pipeline)
    test_pipeline = Compose(pipeline_cfg)

    data = test_pipeline(data)

    # data['keypoint']: (num_clips, M, T, V, C) -> add batch dim -> (1, num_clips, M, T, V, C)
    keypoint = data['keypoint']
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.FloatTensor(keypoint)
    keypoint = keypoint.unsqueeze(0).to(device)

    with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
        with torch.no_grad():
            scores = model(keypoint, return_loss=False)[0] # numpy (num_classes,)
        returned_features = h.layer_outputs if outputs else None
    
    score_tuples = tuple(zip(range(len(scores)), scores.tolist()))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top_k_label = score_sorted[:top_k]

    if outputs:
        return top_k_label, returned_features

    return top_k_label


@torch.no_grad()
def inference_similarity(model, keypoint):
    """
    Args:
        model (nn.Module): The loaded recognizer.
        keypoint : (N, M, T, V, C) tensor, preprocessed input data.
    
    Returns:
        probs: (N, num_classes) softmax probs
        pred: (N,) Predicted class indice
    """
    model.eval()
    device = next(model.parameters()).device
    keypoint = keypoint.to(device)

    # z query
    _, z_query = model.backbone(keypoint) # (N, V*V)

    # projection
    f_query = model.cls_head.csc_loss.cl_fc(z_query) # (N, 256)
    f_query = F.normalize(f_query, p=2, dim=1) # L2 normalization

    # memory bank m_k
    # avg_f : (256, num_classes)
    m = model.cls_head.avg_f.to(device) # (h, K)
    m = m.T  # (K, h)
    m = F.normalize(m, p=2, dim=1) # L2

    tau = model.cls_head.csc_loss.tmp # temperature

    # cosine similarity
    scores = (m @ f_query.T).T / tau
    probs = F.softmax(scores, dim=1)
    pred = probs.argmax(dim=1)

    return probs, pred




