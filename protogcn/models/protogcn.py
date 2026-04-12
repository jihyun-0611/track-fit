import copy as cp
import torch
import torch.nn as nn

from ..utils import Graph
from .mte import MTE
from .mstcn import TCN, MSTCN

EPS = 1e-4


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        self.mte = MTE(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride==1):
            self.residual = lambda x: x
        else: # 1x1 conv for same channels 
            self.residual = TCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        residual = self.residual(x)
        x, gcl_graph = self.mte(x, A)
        x = self.tcn(x) + residual
        return self.relu(x), gcl_graph


class PrototypeReconstructionNetwork(nn.Module):
    '''
    Prototype Reconstruction Network

    '''
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(dim, n_prototype, bias=False)
        self.memory = nn.Linear(n_prototype, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        r = self.softmax(self.query(x))
        z = self.memory(r)
        return self.dropout(z)


class ProtoGCN(nn.Module):
    def __init__(self,
                 graph_cfg, 
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2, 
                 num_stages =10,
                 inflate_stages=[5, 8], 
                 down_stages=[5, 8], 
                 data_bn_type='VC',
                 num_person = 2,
                 num_prototype=100,
                 pretrained=None,
                 **kwargs):
        '''
        Args:
            graph_cfg: (layout, mode) for skeleton graph information
            base_channels: initial output channels  
            ch_ratio: increasing ratio for channels
            num_stages: num of GCN BLOCK
            inflate_stages: where to increase channels
            down_stages: where to decrease time (T) channels
            data_bn_type: 
                    'VC'(in_channels*num_nodes) or 'MVC'(num_person*in_channels*num_nodes)
        '''
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCNBlock(self.in_channels, self.base_channels, A.clone(), 1, residual=False)]
        
        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages+1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCNBlock(in_channels, out_channels, A.clone(), stride))
            down_times += (i in down_stages)
        out_channels = base_channels

        if self.in_channels == self.base_channels:
            num_stages -= 1 # do not have first block

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

        self.post = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        dim = 384 # base_channels * 4
        self.prn = PrototypeReconstructionNetwork(dim, num_prototype)

    def init_weights(self):
        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained, weights_only=False), strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # Data batch normalization
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M*V*C, T))
        else:
            x = self.data_bn(x.view(N*M, V*C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N*M, C, T, V)

        # GCN_Block forward
        graph_list= []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            graph_list.append(gcl_graph) # N*M C V V

        # N C T V -> N M C T V
        x = x.reshape((N, M) + x.shape[1:])
        grp_channels = x.size(2)

        # input for PRN
        
        last_graph = graph_list[-1]
        # flatten : N C V V -> N C V*V
        last_graph = last_graph.view(N, M, grp_channels, V, V).mean(1).view(N, grp_channels, V*V)

        features = last_graph.permute(0, 2, 1) # N V*V C
        recon_graph = self.prn(features) # N V*V C
        batch_reconstructed = recon_graph.permute(0, 2, 1).view(N, grp_channels, V, V)

        # N C V V
        batch_reconstructed = self.post(batch_reconstructed)
        batch_reconstructed = self.relu(self.bn(batch_reconstructed))
        # N V*V
        output_graph = batch_reconstructed.mean(1).view(N, -1)

        return x, output_graph



        

