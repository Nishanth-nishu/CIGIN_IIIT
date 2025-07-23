import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import Set2Set
from math import sqrt

class GraphTransformerLayer(nn.Module):
    """Custom Graph Transformer Layer compatible with all DGL versions"""
    def __init__(self, node_dim=42, num_heads=4, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        # Projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        
        # Edge feature handling
        self.edge_proj = nn.Linear(10, num_heads)  # edge_dim=10
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)
        
    def forward(self, g, h, e_feat=None):
        # Projections
        q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(-1, self.num_heads, self.head_dim)
        
        # Attention scores
        attn_scores = torch.einsum('nhd,mhd->nhm', q, k) / sqrt(self.head_dim)
        
        # Add edge-based attention biases
        if e_feat is not None:
            edge_bias = self.edge_proj(e_feat).permute(1,0)  # [num_heads, E]
            g.edata['b'] = edge_bias
            g.update_all(fn.copy_e('b', 'm'), fn.sum('m', 'b_sum'))
            attn_scores = attn_scores + g.ndata['b_sum'].permute(1,0,2)  # [N, N, H]
        
        # Masking based on adjacency
        adj = g.adjacency_matrix().to_dense().bool()
        attn_scores = attn_scores.masked_fill(~adj.unsqueeze(-1), float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Output
        out = torch.einsum('nhm,mhd->nhd', attn_weights, v)
        out = out.reshape(-1, self.node_dim)
        out = self.out_proj(out)
        
        # Residual + norm
        return self.norm(h + self.dropout(out))

class GraphTransformerGather(nn.Module):
    def __init__(self, node_dim=42, edge_dim=10, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(node_dim) 
            for _ in range(num_layers)
        ])
        
    def forward(self, g, n_feat, e_feat=None):
        h = n_feat
        for layer in self.layers:
            h = layer(g, h, e_feat)
        return h

class CIGINModel(nn.Module):
    def __init__(self, interaction='dot'):
        super().__init__()
        self.interaction = interaction
        
        # Graph Transformer encoders
        self.solute_encoder = GraphTransformerGather(node_dim=42)
        self.solvent_encoder = GraphTransformerGather(node_dim=42)
        
        # Prediction layers
        self.fc1 = nn.Linear(8 * 42, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)
        
    def forward(self, data):
        solute, solvent, solute_len, solvent_len = data
        
        # Get Transformer features
        solute_feats = self.solute_encoder(
            solute, solute.ndata['x'].float(), 
            solute.edata['w'].float() if 'w' in solute.edata else None
        )
        solvent_feats = self.solvent_encoder(
            solvent, solvent.ndata['x'].float(),
            solvent.edata['w'].float() if 'w' in solvent.edata else None
        )
        
        # Interaction phase (preserves interpretability)
        len_map = torch.mm(solute_len.t(), solvent_len)
        
        if 'dot' in self.interaction:
            interaction_map = torch.tanh(
                torch.mm(solute_feats, solvent_feats.t()) / sqrt(42))
            interaction_map = torch.mul(len_map.float(), interaction_map)
        
        # Cross-message passing
        solvent_prime = torch.mm(interaction_map.t(), solute_feats)
        solute_prime = torch.mm(interaction_map, solvent_feats)
        
        # Prediction phase
        solute_out = torch.cat([solute_feats, solute_prime], dim=1)
        solvent_out = torch.cat([solvent_feats, solvent_prime], dim=1)
        
        # Pooling
        solute_pool = solute_out.mean(dim=0, keepdim=True)
        solvent_pool = solvent_out.mean(dim=0, keepdim=True)
        
        # Final prediction
        out = torch.cat([solute_pool, solvent_pool], dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out, interaction_map
