import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_softmax


class RelationGraphEncoder(nn.Module):
    """关系图编码器 (严格遵循因果图聚合公式)"""

    def __init__(self, n_relations, hidden_dim, num_layers=2):
        super().__init__()
        self.n_relations = n_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 默认的关系嵌入层
        self.embedding = nn.Embedding(n_relations, hidden_dim)

        self.layers = nn.ModuleList([
            CausalRelationLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, q_rel, edge_index):
        # 根据查询关系 q_rel 动态初始化嵌入层
        # 创建一个与嵌入矩阵相同形状的全零矩阵
        embeddings = torch.zeros(self.n_relations, self.hidden_dim).cuda()

        # 对应于查询关系 q_rel 的位置设为 1
        embeddings[q_rel] = 1.0

        # 使用这种自定义的嵌入矩阵作为权重
        self.embedding.weight.data.copy_(embeddings)

        # 继续执行因果图聚合
        h = self.embedding.weight
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class CausalRelationLayer(nn.Module):
    """实现因果图聚合公式的GNN层"""

    def __init__(self, hidden_dim):
        super().__init__()
        # 公式中的参数初始化
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 关系特征变换矩阵
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)  # 注意力计算向量
        self.W1 = nn.Linear(hidden_dim, hidden_dim)  # 邻居聚合变换矩阵
        self.W2 = nn.Linear(hidden_dim, hidden_dim)  # 自环变换矩阵
        #self.act = nn.ReLU()
        self.act = nn.LeakyReLU()

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, h, edge_index):
        if edge_index.size(1) == 0:
            return h

        # 确保计算在相同设备
        edge_index = edge_index.to(h.device)
        src, dst = edge_index  # 获取边连接的源节点和目标节点

        # Step 1: 计算因果注意力权重
        h_trans = self.W(h)  # 应用特征变换 [N, hidden_dim]

        # 拼接变换后的特征 (公式中的[Wh_ru || Wh_rv])
        edge_features = torch.cat([h_trans[src], h_trans[dst]], dim=1)  # [E, 2*hidden_dim]

        # 计算注意力分数 (公式中的a^T[...])
        attn_scores = self.a(edge_features).squeeze()  # [E]

        # 使用scatter_softmax进行归一化
        attn_weights = scatter_softmax(attn_scores, dst, dim=0)  # [E]

        # Step 2: 分离自环边和非自环边
        mask = src != dst  # 非自环边掩码
        self_loop_mask = src == dst  # 自环边掩码

        # Step 3: 邻居聚合 (公式第一项)
        neighbor_src = src[mask]
        neighbor_dst = dst[mask]
        neighbor_weights = attn_weights[mask]

        # 聚合邻居消息
        neighbor_msg = neighbor_weights.unsqueeze(-1) * h[neighbor_src]
        aggregated_neighbor = scatter(neighbor_msg, neighbor_dst,
                                      dim=0, dim_size=h.size(0))  # [N, hidden_dim]

        # Step 4: 自环项处理 (公式第二项)
        self_loop_weights = attn_weights[self_loop_mask]
        self_loop_nodes = dst[self_loop_mask]
        self_term = self_loop_weights.unsqueeze(-1) * h[self_loop_nodes]

        # Step 5: 组合并应用变换 (公式中的W1和W2)
        h_neighbor = self.W1(aggregated_neighbor)
        h_self = self.W2(self_term)

        # 残差连接并激活
        # h_new = self.act(h_neighbor + h_self)
        h_new = self.act(h_neighbor + h_self)
        return h_new


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx, rel_emb):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = rel_emb[rel]  # 使用关系编码器的输出

        r_idx = edges[:, 0]
        h_qr = rel_emb[q_rel][r_idx]

        message = hs + hr

        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class RED_GNN_induc(torch.nn.Module):
    def __init__(self, params, loader):
        super(RED_GNN_induc, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        # 关系编码器
        self.relation_encoder = RelationGraphEncoder(
            n_relations=loader.total_relations,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers
        )

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='train'):
        # Set loader's mode to get corresponding relation graph
        relation_edges = self.loader.set_mode(mode)

        n = len(subs)
        n_ent = self.loader.n_ent

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        # Pass the mode-specific relation edges
        rel_emb = self.relation_encoder(q_rel, relation_edges)

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        for i in range(self.n_layer):
            # Get neighbors using the current mode
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx, rel_emb)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all