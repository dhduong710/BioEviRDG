import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_softmax


class RelationGraphEncoder(nn.Module):
    """Relation graph encoder with multi-head attention causal message passing"""

    def __init__(self, n_relations, hidden_dim, num_layers=2, num_heads=8):
        super().__init__()
        self.n_relations = n_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Default relation embedding layer
        self.embedding = nn.Embedding(n_relations, hidden_dim)

        # Multi-head attention causal message passing layers
        self.layers = nn.ModuleList([
            MultiHeadCausalRelationLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, q_rel, edge_index):
        # Dynamically initialize embeddings based on query relation q_rel
        # Create a zero matrix of the same shape as the embedding matrix
        embeddings = torch.zeros(self.n_relations, self.hidden_dim).cuda()

        # Set the position corresponding to query relation q_rel to 1
        embeddings[q_rel] = 1.0

        # Use this custom embedding matrix as weights
        self.embedding.weight.data.copy_(embeddings)

        # Perform causal graph aggregation
        h = self.embedding.weight
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class MultiHeadCausalRelationLayer(nn.Module):
    """
    Multi-head attention causal message passing for relation graphs
    
    Implements equation (3) from the paper:
    h^ℓ_rv|rq = σ((1/H) * ∑^H_h=1 W^h_O [W^h_1 * ∑_ru∈Npast(rv) α^h_rurv * h^(ℓ-1)_ru|rq + W^h_2 * α^h_rv,rv * h^(ℓ-1)_rv|rq])
    """
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Parameter matrices for each head
        self.W1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        self.W2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        self.WO = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        
        # Attention projection matrix (shared across heads)
        self.W_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Attention vectors for each head
        self.a = nn.ParameterList([nn.Parameter(torch.Tensor(2 * hidden_dim)) for _ in range(num_heads)])
        
        # Activation function
        self.act = nn.ReLU()
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_attn.weight)
        for head in range(self.num_heads):
            nn.init.xavier_uniform_(self.W1[head].weight)
            nn.init.xavier_uniform_(self.W2[head].weight)
            nn.init.xavier_uniform_(self.WO[head].weight)
            # 对一维 attention 向量采用正态分布初始化
            nn.init.normal_(self.a[head], mean=0.0, std=0.1)

    
    def forward(self, h, edge_index):
        """
        Forward pass implementing the multi-head attention aggregation
        
        Args:
            h: Relation embeddings [num_relations, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated relation embeddings [num_relations, hidden_dim]
        """
        if edge_index.size(1) == 0:
            return h
            
        device = h.device
        edge_index = edge_index.to(device)
        src, dst = edge_index  # Source and target nodes
        
        # Project node features for attention computation
        h_attn = self.W_attn(h)  # [num_relations, hidden_dim]
        
        # Initialize results tensor to accumulate outputs from each head
        results = torch.zeros_like(h)
        
        # Process each attention head
        for head in range(self.num_heads):
            # 1. Compute attention weights according to equation (4)
            # Concatenate features for each edge
            edge_features = torch.cat([h_attn[src], h_attn[dst]], dim=1)  # [num_edges, 2*hidden_dim]
            
            # Compute attention scores using the attention vector
            attn_scores = torch.sum(edge_features * self.a[head].unsqueeze(0), dim=1)  # [num_edges]
            
            # Normalize using softmax grouped by target node
            attn_weights = scatter_softmax(attn_scores, dst, dim=0)  # [num_edges]
            
            # 2. Aggregate messages from neighbors
            # Multiply source node features by attention weights
            weighted_msgs = attn_weights.unsqueeze(-1) * h[src]  # [num_edges, hidden_dim]
            
            # Sum messages for each target node
            aggregated_msgs = scatter(weighted_msgs, dst, dim=0, dim_size=h.size(0))  # [num_relations, hidden_dim]
            
            # 3. Apply W1 transformation to aggregated messages
            term1 = self.W1[head](aggregated_msgs)  # [num_relations, hidden_dim]
            
            # 4. Self-loop term: W2 * α_rv,rv * h_rv
            # Find self-loops
            self_loop_mask = src == dst
            self_nodes = src[self_loop_mask]
            self_weights = attn_weights[self_loop_mask]
            
            # Initialize self term
            term2 = torch.zeros_like(term1)
            
            # Apply self-attention where self-loops exist
            if self_loop_mask.any():
                term2[self_nodes] = self.W2[head](self_weights.unsqueeze(-1) * h[self_nodes])
            
            # 5. Combine terms and apply output projection
            combined = term1 + term2
            head_output = self.WO[head](combined)  # [num_relations, hidden_dim]
            
            # Accumulate result
            results += head_output
        
        # Average over heads (1/H term in equation)
        results = results / self.num_heads
        
        # Apply activation function
        return self.act(results)


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

        # Relation encoder with multi-head attention
        self.relation_encoder = RelationGraphEncoder(
            n_relations=loader.total_relations,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads  # Can be parameterized if needed
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