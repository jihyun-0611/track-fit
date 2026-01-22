import torch
import torch.nn as nn

from losses import ClassSpecificContrastiveLoss
from utils import top_k_accuracy

class Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 joint_cfg='coco_new',
                 weight=0.3,
                 dropout=0.0,
                 init_std=0.01):
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
        self.ce_loss = nn.CrossEntropyLoss()

        if joint_cfg == 'coco_new':
            n_channel = 400 # 20*20
        elif joint_cfg == 'nturgb+d':
            n_channel = 625 # 25*25
        else:
            raise ValueError(f"Unknown joint_cfg: {joint_cfg}")

        self.csc_loss = ClassSpecificContrastiveLoss(num_classes, n_channel)
        self.init_weights()


    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, std=self.init_std)
        if self.fc_cls.bias is not None:
            nn.init.constant_(self.fc_cls.bias, 0)


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

        
    def loss(self, cls_score, get_graph, label):
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
        if cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(
                cls_score.detach().cpu().numpy(),
                label.detach().cpu().numpy(),
                topk=(1, 5)
            )
            losses['top1_acc'] = torch.tensor(top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(top_k_acc[1], device=cls_score.device)
        
        # cross-entropy 
        loss_ce = self.ce_loss(cls_score, label)

        # class-specific contrastive loss
        loss_csc = self.csc_loss(get_graph, label.detach(), cls_score.detach())

        # total loss
        total_loss = loss_ce + self.weight * loss_csc.mean()
        losses['loss_cls'] = total_loss

        return losses



        

    
