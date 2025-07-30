import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=6, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.edge_encoder = nn.Linear(10, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, g, node_feat, edge_feat=None):
        N = node_feat.size(0)
        Q = self.q_proj(node_feat).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(node_feat).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(node_feat).view(N, self.num_heads, self.head_dim)

        scores = torch.einsum('ihd,jhd->ijh', Q, K) / math.sqrt(self.head_dim)

        if edge_feat is not None:
            src, dst = g.edges()
            edge_bias = self.edge_encoder(edge_feat).view(-1, self.num_heads)
            bias_matrix = torch.zeros(N, N, self.num_heads, device=scores.device)
            bias_matrix[src, dst] = edge_bias
            scores += bias_matrix

        adj_mask = torch.zeros(N, N, device=scores.device)
        if g.number_of_edges() > 0:
            src, dst = g.edges()
            adj_mask[src, dst] = 1
            adj_mask[dst, src] = 1
        adj_mask.fill_diagonal_(1)

        scores = scores.masked_fill(adj_mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum('ijh,jhd->ihd', attn_weights, V).contiguous().view(N, -1)
        out = self.out_proj(out)
        out = self.norm1(out + node_feat)
        out = self.norm2(out + self.ffn(out))
        return out


class GraphTransformerGatherModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_heads=6, steps=6):
        super(GraphTransformerGatherModel, self).__init__()
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(steps)
        ])

    def forward(self, g, node_feat, edge_feat):
        h = F.relu(self.input_proj(node_feat))
        init = h.clone()
        for layer in self.layers:
            h = layer(g, h, edge_feat)
        return h + init


class GraphTransformerPooling(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=6):
        super(GraphTransformerPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, in_dim))
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, g, node_feat):
        batch_sizes = g.batch_num_nodes().tolist() if hasattr(g, 'batch_num_nodes') else [node_feat.size(0)]
        outputs = []
        start = 0
        for size in batch_sizes:
            segment = node_feat[start:start + size].unsqueeze(0)
            q = self.query.expand(1, -1, -1)
            pooled, _ = self.attn(q, segment, segment)
            outputs.append(pooled.squeeze(0).squeeze(0))
            start += size
        return self.proj(torch.stack(outputs, 0))


class CIGINGraphTransformerModel(nn.Module):
    def __init__(self, node_dim=42, edge_dim=10, heads=6, steps=6):
        super(CIGINGraphTransformerModel, self).__init__()
        self.solute_gather = GraphTransformerGatherModel(node_dim, edge_dim, node_dim, heads, steps)
        self.solvent_gather = GraphTransformerGatherModel(node_dim, edge_dim, node_dim, heads, steps)

        self.solute_pool = GraphTransformerPooling(2 * node_dim, 4 * node_dim, heads)
        self.solvent_pool = GraphTransformerPooling(2 * node_dim, 4 * node_dim, heads)

        self.imap = nn.Linear(2 * node_dim, 1)
        self.fc1 = nn.Linear(8 * node_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, data):
        solute, solvent, len_solu, len_solvent = data
        h_solu = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        h_solv = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())

        interaction = torch.mm(h_solu, h_solv.t())
        interaction = torch.tanh(interaction)

        solute_prime = torch.mm(interaction, h_solv)
        solvent_prime = torch.mm(interaction.t(), h_solu)

        h_solu = torch.cat([h_solu, solute_prime], dim=1)
        h_solv = torch.cat([h_solv, solvent_prime], dim=1)

        pooled_solu = self.solute_pool(solute, h_solu)
        pooled_solv = self.solvent_pool(solvent, h_solv)

        final = torch.cat([pooled_solu, pooled_solv], 1)
        x = F.relu(self.fc1(final))
        x = F.relu(self.fc2(x))
        return self.fc3(x), interaction
