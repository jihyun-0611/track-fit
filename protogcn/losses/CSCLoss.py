import torch
import torch.nn as nn

"""
Class-Spacific Constrastive Learning
"""

class ClassSpecificContrastiveLoss(nn.Module):
    def __init__(self,
                 n_class,
                 n_channel=625,
                 h_channel=256,
                 tmp=0.125, # τ : temperature scaling
                 momentum=0.9,
                 pred_threshold=0.0, 
                 prior_path=None,
                 prior_alpha=1.0,
                 prior_warmup_epochs=5,
                 prior_mode='off'): # 'bayes' | 'margin' | 'off'
        super(ClassSpecificContrastiveLoss, self).__init__()
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.n_class = n_class
        self.tmp = tmp
        self.momentum = momentum
        self.pred_threshold = pred_threshold
        self.register_buffer('avg_f', torch.randn(self.h_channel, self.n_class))
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')

        if prior_path is not None and prior_mode != 'off':
            blob = torch.load(prior_path, map_location='cpu', weights_only=False)
            P = blob['prior']
            assert P.shape == (n_class, n_class), f"prior shape {P.shape} != ({n_class}, {n_class})"
            self.register_buffer('log_prior', torch.log(P.clamp_min(1e-8)))
            if prior_mode == 'margin':
                diag = self.log_prior.diag().unsqueeze(1)
                self.register_buffer('log_prior_norm', self.log_prior - diag)
        else:
            self.register_buffer('log_prior', torch.zeros(n_class, n_class))
        self.prior_alpha = prior_alpha
        self.prior_warmup_epochs = prior_warmup_epochs
        self.prior_mode = prior_mode
        self.current_epoch = 0


    def onehot(self, label):
        """one-hot encoding"""
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)

        return ones.view(*size).float()
    
    def get_mask(self, lbl_one, pred_one, logit):
        '''Only sufficiently high confidence are stored in the memory bank.'''
        # batch num_class
        tp = lbl_one * pred_one
        tp = tp * (logit > self.pred_threshold).float()
        return tp
    
    def local_average(self, f, mask):
        '''
        m_k = alpha * m_k + (1-alpha)f_k

        f : projected feature 
            (batch, h_channel)
        mask : sample mask(label == pred & > threshold)) 
            (batch, num_class) 
        '''
        b, k = mask.size()

        # average by classes

        # (256, batch)
        f = f.permute(1, 0) 
        avg_f = self.avg_f.detach().to(f.device)
        # (batch, num_class) -> (1, num_class)
        mask_sum = mask.sum(0, keepdim=True) 
        # (256 , batch) * (batch, num_class) -> (256 , num_class)
        f_mask = torch.matmul(f, mask) 
        # f̄_k: average of features by classes (256, num_class)
        f_mask = f_mask / (mask_sum + 1e-12) 


        # momentum
        has_object = (mask_sum > 1e-8).float()

        has_object[has_object > 0.1] = self.momentum
        has_object[has_object <= 0.1] = 1.0

        # update memory bank
        # 256 num_class
        f_mem = avg_f * has_object + (1-has_object) * f_mask
        with torch.no_grad():
            self.avg_f.copy_(f_mem)
        
        return f_mem
    
    def get_score(self, feature, lbl_one, f_mem):
        (b, c), k = feature.size(), self.n_class
        
        # L2 Norm

        # batch, h_channel
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True)+1e-12)
        # n_class, h_channel
        f_mem = f_mem.permute(1, 0)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True)+1e-12)

        # consine similarity
        # num_class, batch = (num_class, h_channel) * (h_channel, batch)
        score_mem = torch.matmul(f_mem, feature.permute(1, 0))
        score_cl = score_mem / self.tmp 

        return score_cl
    
    def forward(self, feature, lbl, logit):
        # batch, h_channel
        feature = self.cl_fc(feature)
        # batch, num_classes -> batch
        pred = logit.max(1)[1]

        # batch, num_classes
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)
        # batch, num_classes
        logit = torch.softmax(logit, 1)

        mask = self.get_mask(lbl_one, pred_one, logit)
        f_mem = self.local_average(feature, mask)
        score_cl = self.get_score(feature, lbl_one, f_mem)

        # batch, num_class
        score_cl = score_cl.permute(1, 0).contiguous()

        if self.prior_mode != 'off':
            # linear warmup over [0, warmup_epochs)
            ramp = min(1.0, self.current_epoch / max(1, self.prior_warmup_epochs))
            alpha_eff = self.prior_alpha *ramp
            if self.prior_mode == 'bayes':
                bias = self.log_prior[lbl] # batch, n_class
                score_cl = score_cl + alpha_eff *bias
            elif self.prior_mode == 'margin':
                bias = self.log_prior_norm[lbl] # (batch, n_class), diag=0
                score_cl = score_cl - alpha_eff *bias # subtract: penalty on confused j

        return self.loss(score_cl, lbl).mean()
