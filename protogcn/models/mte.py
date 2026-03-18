# MTE(Motion Topology Enhancement)

import torch
import torch.nn as nn


class MTE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A, 
                 ratio=0.125):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # C' = C/K : multi-head 설정 
        # K = subets의 개수 -> 8개 
        # ratio는 1/8 = 0.125 
        self.num_subsets = A.size(0)
        self.ratio = ratio
        mid_channels = int(self.ratio*out_channels)
        self.mid_channels = mid_channels # C'

        self.A = nn.Parameter(A.clone())

        if in_channels != out_channels:
            self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels))
        else:
            self.residual=lambda x:x

        self.h_last = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1), # mid_channels*num_subsets = output_channels
            nn.BatchNorm2d(mid_channels * self.num_subsets), 
            nn.ReLU(inplace=True)
        )

        self.h_q = nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1)
        self.h_k = nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1)

        self.h_l = nn.Conv2d(mid_channels*self.num_subsets, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

    
    def forward(self, x, A=None):
        # x : features 
        # n c t v
        n, c, t, v = x.shape
        residual = self.residual(x)

        # A : Adjacency Matrix
        # K V V
        A = self.A 
        A = A[None, :, None, None] # 1 k 1 1 V V 

        """Motion Topology Enhancement"""

        # multi-head setting N, K, C', T, V
        # H^(l-1), H^Q, H^K
        h_last = self.h_last(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        h_q, h_k = None, None

        tmp_x = x

        # N K C' T V
        h_q = self.h_q(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        h_k = self.h_k(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        # Average Pooling(T): N K C' 1 V
        h_q = h_q.mean(dim=-2, keepdim=True)
        h_k = h_k.mean(dim=-2, keepdim=True)

        graph_list = []

        # A_inter
        # differnce : T1(H_Q)- T2(H_k) 
        # N K C' 1 V V = N K C' 1 V 1 - N K C' 1 1 V
        diff = h_q.unsqueeze(-1) - h_k.unsqueeze(-2)
        A_inter = self.tanh(diff)
        A_inter = A_inter * self.alpha[0]

        # N K C' 1 V V = N K C' 1 V V + 1 K 1 1 V V
        A = A_inter + A
        graph_list.append(A_inter)

        # A_intra 
        # H_Q(H_K)^T
        # N K C' 1 V * N K C' 1 V = N K 1 1 V V
        A_intra = torch.einsum('nkctv, nkctw -> nktvw', h_q, h_k)[:, :, None]
        A_intra = self.softmax(A_intra)
        A_intra = A_intra * self.beta[0]

        # N K C' 1 V V = N K 1 1 V V + N K C' 1 V V
        A = A_intra + A
        graph_list.append(A_intra)
        # N K C V V
        A = A.squeeze(3)

        # (A_0 + A_intra + A+inter)H^(l-1)
        # N K C' T V = N K C' T V * N K C' V V
        x = torch.einsum('nkctv,nkcvw->nkctw', h_last, A).contiguous()
        # N K C' T V -> N K*C' T V
        x = x.reshape(n, -1, t, v)
        x = self.h_l(x)

        get_gcl_graph = graph_list[0] + graph_list[1]
        # N K C' 1 V V -> N K C' V V
        get_gcl_graph = get_gcl_graph.squeeze(3)
        # N K C' T V -> N K*C' T V
        get_gcl_graph = get_gcl_graph.reshape(n, -1, v, v)
        
        return self.relu(self.bn(x) + residual), get_gcl_graph

