import torch
import torch.nn as nn

from ..losses import ClassSpecificContrastiveLoss
from ..utils import top_k_accuracy

class Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 joint_cfg='coco_new',
                 weight=0.3,
                 dropout=0.0,
                 label_smoothing=0.0,
                 init_std=0.01,
                 prior_path=None, prior_mode='off',
                 prior_hard_topk=3, prior_hard_norm='row_max', 
                 csc_prior_alpha=0.0, csc_prior_warmup_epochs=5,
                 ce_prior_alpha=0.0, ce_prior_warmup_epochs=5,     
                ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weight = weight
        self.init_std = init_std

        self.pool = nn.AdaptiveAvgPool2d(1)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if joint_cfg == 'coco_new':
            n_channel = 400 # 20*20
        elif joint_cfg == 'nturgb+d':
            n_channel = 625 # 25*25
        else:
            raise ValueError(f"Unknown joint_cfg: {joint_cfg}")

        self._build_prior_buffer(prior_path, prior_mode, prior_hard_topk, prior_hard_norm)
        self.csc_loss = ClassSpecificContrastiveLoss(num_classes, n_channel)

        self.csc_prior_alpha = csc_prior_alpha
        self.csc_prior_warmup_epochs = csc_prior_warmup_epochs

        self.ce_prior_alpha = ce_prior_alpha
        self.ce_prior_warmup_epochs = ce_prior_warmup_epochs

        self.register_buffer('_current_epoch', torch.tensor(0))

        self.init_weights()


    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, std=self.init_std)
        if self.fc_cls.bias is not None:
            nn.init.constant_(self.fc_cls.bias, 0)


    def _build_prior_buffer(self, path, mode, topk, norm):
        C = self.num_classes
        if path is None or mode == 'off':
            self.register_buffer('W', torch.zeros(C, C))
            self.prior_mode = 'off'
            return
        
        blob = torch.load(path, map_location='cpu', weights_only=False)
        P = blob['prior'].float()
        
        assert P.shape == (C, C)
        
        if mode == 'hard_margin':
            if 'confusion_raw' not in blob:
                raise KeyError(
                    "hard_margin mode requires 'confusion_raw' in prior file. "
                    "Regenerate it with protogcn.tools.build_confusion_prior."
                )
            C_raw = blob['confusion_raw'].float()
            assert C_raw.shape == (C, C)
            W = self._make_hard_margin_W(C_raw, topk, norm)
        elif mode == 'bayes':
            W = torch.log(P.clamp_min(1e-8))   # 기존 log_prior와 동등
        elif mode == 'margin':
            log_P = torch.log(P.clamp_min(1e-8))
            W = log_P.diag().unsqueeze(1) - log_P
        else:
            raise ValueError(f'Unknown prior_mode: {mode}')
        self.register_buffer('W', W)
        self.prior_mode = mode


    def _make_hard_margin_W(self, P, topk, norm):
        W = P.clone()
        W.fill_diagonal_(0.0)

        if topk is not None and topk > 0:
            k = min(int(topk), self.num_classes - 1)
            vals, idx = torch.topk(W, k=k, dim=1)
            W_topk = torch.zeros_like(W)
            W_topk.scatter_(1, idx, vals)
            W = W_topk

        if norm == 'row_max':
            denom = W.max(dim=1, keepdim=True).values.clamp_min(1e-8)
            W = W / denom
        elif norm == 'row_sum':
            denom = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
            W = W / denom
        elif norm in ('none', None):
            pass
        else:
            raise ValueError(f'Unknown prior_hard_norm: {norm}')

        W.fill_diagonal_(0.0)
        return W


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features from backbone
                Shape: (N, M, C, T, V) for GCN mode.
        
        Returns:
            torch.Tensor: Classification scores
                Shape: (N, num_classes)
        """
        N, M, C, T, V = x.shape
        x = x.reshape(N*M, C, T, V)
        x = self.pool(x) # (N*M, C, 1, 1)

        x = x.reshape(N, M, C)
        x = x.mean(dim=1) #(N, C)

        assert x.shape[1] == self.in_channels, f"Input channels mismatch: got {x.shape[1]}, expected {self.in_channels}"

        if self.dropout is not None:
            x = self.dropout(x)
        
        cls_score = self.fc_cls(x)
        return cls_score

        
    def loss(self, cls_score, get_graph, label, compute_acc=False):
        """
        Compute total loss
        
        Args: 
            cls_score(torch.Tensor): Classification scores, shape (N, num_classes)
            get_graph: Graph features from  backbone for CSCL
            label: Ground Truth labels

        Returns:
            dict: Dictionary containing losses and accuracy metrics
        """
        losses = dict()

        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size(0) == self.num_classes and cls_score.size(0) == 1:
            label = label.unsqueeze(0)

        # comput top-k accuracy
        if compute_acc and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(
                cls_score.detach().cpu().numpy(),
                label.detach().cpu().numpy(),
                topk=(1, 5)
            )
            losses['top1_acc'] = torch.tensor(top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(top_k_acc[1], device=cls_score.device)
        
        # cross-entropy with optional confusion-aware additive margin on negatives
        if self.ce_prior_alpha > 0 and self.prior_mode != 'off':
            ramp = min(1.0, self._current_epoch.item() / max(1, self.ce_prior_warmup_epochs))
            cls_score_adj = cls_score + (self.ce_prior_alpha * ramp) * self.W[label]
            loss_ce = self.ce_loss(cls_score_adj, label)
        else:
            loss_ce = self.ce_loss(cls_score, label)

        # class-specific contrastive loss
        cscl_W = None
        if self.csc_prior_alpha > 0 and self.prior_mode != 'off':
            ramp = min(1.0, self._current_epoch.item() / max(1, self.csc_prior_warmup_epochs))
            cscl_W = (self.csc_prior_alpha * ramp) * self.W
        loss_csc = self.csc_loss(get_graph, label.detach(), cls_score.detach(), W=cscl_W)

        # total loss
        total_loss = loss_ce + self.weight * loss_csc.mean()
        losses['loss_cls'] = total_loss

        return losses



        

    
