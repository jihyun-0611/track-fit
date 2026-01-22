import torch
import torch.nn as nn


class TCN(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size=9,
                 stride=1,
                 dilation=1,
                 dropout=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        pad = (kernel_size + (kernel_size-1)*(dilation-1)-1) // 2

        # conv1d (T)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0), 
            stride=(stride, 1),
            dilation=(dilation, 1))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))
    
    def init_weights(self):
        # init conv
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')
        nn.init.constant_(self.conv.bias, 0)
        # init bn
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


class MSTCN(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 mid_channels=None, 
                 num_joints=20,
                 dropout=0,
                 # (kernel, dilation), (max pooling, kernel), (1x1 conv)
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.num_joints = num_joints
        self.add_coef = nn.Parameter(torch.zeros(self.num_joints))

        # set branches' out channels
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels-mid_channels *(num_branches-1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        # create net with branch
        branches = []
        for i, cfg in enumerate(ms_cfg):
            b_out_channel = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(in_channels, b_out_channel, kernel_size=1, stride=1)
                )
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, b_out_channel, kernel_size=1),
                        nn.BatchNorm2d(b_out_channel), 
                        self.relu,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))
                    )
                )
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, b_out_channel, kernel_size=1),
                    nn.BatchNorm2d(b_out_channel),
                    self.relu,
                    TCN(b_out_channel, kernel_size=cfg[0], stride=stride, dilation=cfg[1])
                )
            )

        self.branches = nn.ModuleList(branches)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(mid_channels*(num_branches-1)+rem_mid_channels), 
            self.relu,
            nn.Conv2d(
                mid_channels*(num_branches-1)+rem_mid_channels,
                out_channels,
                kernel_size=1
            )
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def _forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for seq in self.branches:
            out = seq(x)
            branch_outs.append(out)
        
        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
         # (N,C,T,1) * (V,) → (N,C,T,V)
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coef[:V])

        features = local_feat + global_feat
        return self.transform(features)
    
    def forward(self, x):
        out = self._forward(x)
        out = self.bn(out)
        return self.drop(out)
