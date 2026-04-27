import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops

from torch_scatter import scatter_add

from math import ceil

"""
GWN 模拟graph structure中node features随时间的传播过程--similar to Wave PDEs，从而捕捉节点在graph spatial的dynamic behabors。
- compared to GCNs, 它引入temporal propagation process (多步更新)，并允许higher-order propagation and modeling.
- GWN model
    - WavePDEFunc (解决图波动方程)
        - WaveConv: 多层传播卷积
        - lin0, lin1: 初始状态的2 linear mapping.
        - exp_conv_0: 特殊初始化传播 (Euler显示)
- Linear layer for final classification. 

最大区别：GCN and GWN
- GCN：只传播一次 (即one step)，本质是低阶近似。
- GWN: 传播多步，模拟节点特征随时间扩散(or 波动)。
这对筋膜又时序特性或传播延迟的graph data (e.g., soical messages, disaster diffusion)非常重要。
"""
class GWN(nn.Module):
    def __init__(self, feature_size, hid_dim, num_relations, time=1., dropout=0.5,
                 laplacian='sym', method='exp', dt=1., init_residual=False):
        super(GWN, self).__init__()

        self.dropout = dropout

        # GWN pde, simulate 时间演化 PDEs
        # self.pde = WavePDEFunc(dataset, dim_hidden, time, dt, dropout, laplacian, method, init_residual)
        self.pde_layers = nn.ModuleList([WavePDEFunc(feature_size, hid_dim, time, dt, dropout, laplacian, method, init_residual)
                                         for _ in range(num_relations)])  # DHAN-4: original version

        # Linear layer for classification
        # self.lin = nn.Linear(dim_hidden, dataset.num_classes)

    def reset_parameters(self):
        self.pde.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []

        features = features.to(device)

        for i, (edge_index) in enumerate(multi_r_data):
            x = F.dropout(features, p=self.dropout, training=self.training)
            # ---------------2 GCN layers from FinEvent h1-------------------------------------
            gcn_embedding = self.pde_layers[i](x, edge_index)  # (100,256)
            gcn_embedding = F.dropout(gcn_embedding, p=self.dropout, training=self.training)

            embed_list.append(torch.unsqueeze(gcn_embedding[batch_nodes], dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        # final_embed, att_val = self.simpleAttnLayer(multi_embed, device)  # (100, 64)
        final_embed = multi_embed.view(batch_nodes.shape[0], -1)
        del multi_embed
        gc.collect()

        return final_embed

        # x = F.dropout(x, p=self.dropout, training=self.training)
        # # Solve the initial value problem of graph wave equation
        # x = self.pde(x, edge_index)  # PDE solution: graph node representation随时间传播
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # Finial linear transformation
        # x = self.lin(x)  # 分类，no, I need node representation.
        # return x

"""
# graph wave PDE 求解器
- 改模块目的是模拟wave equation的离散时间迭代形式，x^{t+1} = Lx^t - x^{t-1}.
- 其初始化形式：x^1 = Lx^0 + Δt * x^0.
- lin0, lin1：map initial input x -> initial state x_0, x_1.
- self.n: 传播时间步数 = total_time/dt
- each step 由WaveConv执行传播过程：x_next = Lx - x_pre.
"""
class WavePDEFunc(nn.Module):
    def __init__(self, input_dim, channels, time, dt, dropout,
                 laplacian='sym', method='exp', init_residual=False):
        super(WavePDEFunc, self).__init__()

        self.time = time
        self.dt = dt
        self.n = ceil(time / dt)
        self.dropout = dropout

        self.laplacian = laplacian
        self.method = method
        self.init_residual = init_residual

        # GWN convolutional layers
        self.convs = nn.ModuleList([
            WaveConv(channels, dt, dropout, laplacian, method, init_residual)
            for _ in range(self.n - 1)])

        if method == 'exp':
            self.exp_conv_0 = WaveConv(channels, dt, dropout, laplacian, method, init_residual, exp_init=True)

        # Linear layers
        self.lin0 = nn.Linear(input_dim, channels)
        self.lin1 = nn.Linear(input_dim, channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.method == 'exp':
            self.exp_conv_0.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

    def set_init_value(self, x, edge_index):
        phi_0 = self.lin0(x)
        phi_1 = self.lin1(x)

        if self.method == 'exp':
            x_1 = self.exp_conv_0(phi_1, phi_0, edge_index)
        else:
            raise NotImplementedError

        return phi_0, x_1

    def forward(self, x, edge_index):
        # Set initial value
        x_0, x_1 = self.set_init_value(x, edge_index)

        # Solve the initial value problem of graph wave equation
        x, x_pre = x_1, x_0
        for tn in range(self.n - 1):
            if not self.init_residual:
                x_next = self.convs[tn](x, x_pre, edge_index)
            else:
                x_next = self.convs[tn](x, x_pre, edge_index, x_0=x_0)

            x_pre = x
            x = x_next

        return x

"""
WaveConv: 波动传播卷积 Wave Propagation Convolution.
- Math formula: 模拟PDE中的离散化更新公式 (Euler显示)： x^{t+1} = Lx^t - x^{t-1}.
    - input: x-当前状态；x_pre: previous_step_state; x_0: initial_state (用于residual design).
    - get_laplacian(): 根据设置的laplacian，计算graph structure的laplace weight matrix. 支持sys--归一化 Laplace，fa--feature-aware attention.
    - explicit_propagate(): 使用显示Euler法 (forward Euler)计算: x^{t+1} = propagate(L, x^t) - x^{t-1}
    - message(): PyG的消息传递函数，m_{i <- j} = edge_weight * x_j.
"""
class WaveConv(MessagePassing):
    def __init__(self, channels, dt, dropout,
                 laplacian='sym', method='exp', init_residual=False, exp_init=False):
        super(WaveConv, self).__init__()

        self.channels = channels
        self.dt = dt
        self.dropout = dropout

        self.laplacian = laplacian
        self.method = method
        self.init_residual = init_residual
        self.exp_init = exp_init

        if laplacian == 'fa':
            self.att_j = nn.Linear(channels, 1, bias=False)
            self.att_i = nn.Linear(channels, 1, bias=False)

        if method == 'exp':
            self.scheme = self.explicit_propagate
        else:
            raise NotImplementedError

        if init_residual:
            self.eps = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.laplacian == 'fa':
            self.att_j.reset_parameters()
            self.att_i.reset_parameters()

        if self.init_residual:
            uniform_(self.eps)

    def forward(self, x, x_pre, edge_index, x_0=None):
        edge_index, edge_weight = self.get_laplacian(x, edge_index)
        x = self.scheme(x, x_pre, edge_index, edge_weight)

        if self.init_residual and x_0 is not None:
            x = x + self.eps * x_0

        return x

    def get_laplacian(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)
        num_nodes = x.size(0)

        if self.laplacian == 'sym':
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0., num_nodes)
            edge_index, edge_weight = self.normalize_laplacian(edge_index, edge_weight, num_nodes)

        elif self.laplacian == 'fa':
            edge_index, edge_weight = gcn_norm(edge_index, None, num_nodes, dtype=x.dtype)

            alpha_j = self.att_j(x)[edge_index[0]]
            alpha_i = self.att_i(x)[edge_index[1]]

            alpha = torch.tanh(alpha_j + alpha_i).squeeze(-1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            edge_weight = alpha * edge_weight

        else:
            raise NotImplementedError

        return edge_index, edge_weight

    @staticmethod
    def normalize_laplacian(edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        idx = col
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def explicit_propagate(self, x, x_pre, edge_index, edge_weight):
        num_nodes = x.size(0)

        if self.exp_init:
            edge_weight = self.dt ** 2 * edge_weight / 2
            edge_weight[-num_nodes:] += 1

            x = self.dt * x + self.propagate(edge_index, x=x_pre, edge_weight=edge_weight)

        else:
            edge_weight = self.dt ** 2 * edge_weight
            edge_weight[-num_nodes:] += 2

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight) - x_pre

        return x

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'(channels={self.channels}, dt={self.dt}, laplacian={self.laplacian}, method={self.method})'
