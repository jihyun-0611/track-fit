import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Union, Tuple


class Recognizer(nn.Module):
    """
    Args:
        backbone(nn.Module): Backbone network instance.
        cls_head(nn.Module, optional): Classification head instance.
        train_cfg(dict): Config for training. Default: {}
        test_cfg(dict): Config for testing. Default: {}
    """
    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

        self.train_cfg = train_cfg if train_cfg is not None else {}
        self.test_cfg = test_cfg if test_cfg is not None else {}

        self.init_weights()

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None
    
    def init_weights(self):
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()
        if self.with_cls_head and hasattr(self.cls_head, 'init_weights'):
            self.cls_head.init_weights()

    def extract_feat(self, keypoint):
        """
        Extract features trough backbone
        
        Args: 
            keypoint: Input skeleton data.

        Returns:
            Tuple of (features, graph_info) from backbone.
        """
        return self.backbone(keypoint)
    
    def average_clip(self, cls_score):
        """
        Averaging class score over multiple clips.
        
        Args: 
            cls_score: Class score tensor of shape (batch, num_clips, num_classes).

        Returns:
            Averaged class score of shape (batch, num_classes).
        """
        assert len(cls_score.shape) == 3
        average_clips = self.test_cfg.get('average_clips', 'prob')

        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. Supported: ["score", "prob", None]')

        if average_clips is None:
            return cls_score
        elif average_clips == 'prob':
            return F.softmax(cls_score, dim=2).mean(dim=1)
        else: # 'score'
            return cls_score.mean(dim=1)
        

    def forward_train(self, keypoint, label, **kwargs):
        """
        Forward for training
        
        Args: 
            keypoint: Input skeleton data of shape (N, 1, M, T, V, C).
            label: Ground truth labels.

        Returns: 
            Dict containing loss values.
        """
        assert self.with_cls_head
        assert keypoint.shape[1] == 1 # num_clips ==1 
        keypoint = keypoint[:, 0] # Remove num_clips dimension : (N, M, T, V, C)

        losses = {}
        x, get_graph = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, get_graph, gt_label, **kwargs)
        losses.update(loss)

        return losses
    
    def forward_test(self, keypoint, **kwargs):
        """
        Forward for testing
        
        Args: 
            keypoint: Input skeleton data of shape (N, num_clips, M, T, V, C).

        Returns: 
            Predicted class scores as numpy array.
        """
        assert self.with_cls_head
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:]) # (N*num_clips, M, T, V, C).

        x, get_graph = self.extract_feat(keypoint)

        # Feature extraction mode (for analysis)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)

        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = {'n': 0, 'm': 1, 't': 3, 'v': 4}

            if pool_opt == 'all':
                pool_opt = 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, (tuple, list)):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'

            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight # (num_classes, C)
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv, oc->nmotv', x, w)
                if b is not None:
                    x = x+ b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)
        
        # Normal inference
        cls_score = self.cls_head(x) # (bs * nc, num_classes)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1]) # (bs, nc, num_classes)

        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)

        if isinstance(cls_score, (tuple, list)):
            cls_score = [s.data.cpu().numpy() for s in cls_score]
            return [[s[i] for s in cls_score] for i in range(bs)]
        
        return cls_score.data.cpu().numpy()
    

    def forward(self,
                keypoint,
                label=None,
                return_loss=True,
                **kwargs):
        """
        Define the computation performed at every call.
        
        Args: 
            keypoint: Input skeleton data.
            label: Ground truth labels (required when return_loss=True).
            return_loss: If True, return losses (training mode).
                        If Flase, return predictions (testing mode).

        Returns: 
            Training: Dict of losses
            Testing : Predicted class scores as numpy array.
        """
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)
    
    

        

