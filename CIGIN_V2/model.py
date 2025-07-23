import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphTransformer
from dgl import function as fn

class GraphTransformerGather(nn.Module):
    def __init__(self, node_dim=42, edge_dim=10, num_heads=4, num_layers=3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Graph Transformer layers with residual connections
        self.layers = nn.ModuleList([
            GraphTransformerLayer(node_dim, num_heads, 256, dropout=0.1) 
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        
    def forward(self, g, node_feats, edge_feats=None):
        # Compute attention biases from edge features
        if edge_feats is not None:
            g.edata['a'] = self.edge_proj(edge_feats)  # [E, num_heads]
        
        # Transformer processing
        h = node_feats
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(g, h))
        return h

class GraphTransformerLayer(nn.Module):
    def __init__(self, node_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(node_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, h):
        # Self-attention
        attn_out = self.attention(g, h)
        h = h + self.dropout(attn_out)
        
        # Feed-forward
        return h + self.dropout(self.ffn(h))

class MultiHeadAttention(nn.Module):
    def __init__(self, node_dim, num_heads):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        
    def forward(self, g, h):
        q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim ** 0.5)
        
        # Apply edge-based attention biases if available
        if 'a' in g.edata:
            g.edata['s'] = attn_scores[g.edges()[0], g.edges()[1]] + g.edata['a']
            g.update_all(fn.copy_e('s', 'm'), fn.max('m', 's_max'))
            attn_scores = attn_scores - g.ndata['s_max'].unsqueeze(2)  # Stabilize softmax
        
        attn_weights = F.softmax(attn_scores, dim=1)
        out = torch.einsum('nhm,mhd->nhd', attn_weights, v)
        out = out.reshape(-1, self.node_dim)
        return self.out_proj(out), attn_weights

class CIGINTransformer(nn.Module):
    def __init__(self, interaction='dot'):
        super().__init__()
        self.interaction = interaction
        
        # Graph Transformer encoders
        self.solute_encoder = GraphTransformerGather(node_dim=42, edge_dim=10)
        self.solvent_encoder = GraphTransformerGather(node_dim=42, edge_dim=10)
        
        # Prediction layers
        self.fc1 = nn.Linear(8 * 42, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)
        
    def forward(self, data, return_attention=False):
        solute, solvent, solute_len, solvent_len = data
        
        # Get Transformer features
        solute_feats = self.solute_encoder(solute, solute.ndata['x'].float(), 
                                         solute.edata.get('w', None))
        solvent_feats = self.solvent_encoder(solvent, solvent.ndata['x'].float(), 
                                           solvent.edata.get('w', None))
        
        # Interaction phase (preserves interpretability)
        len_map = torch.mm(solute_len.t(), solvent_len)
        
        if 'dot' in self.interaction:
            interaction_map = torch.tanh(
                torch.mm(solute_feats, solvent_feats.t()) / (42 ** 0.5))
            interaction_map = torch.mul(len_map.float(), interaction_map)
        
        # Cross-message passing
        solvent_prime = torch.mm(interaction_map.t(), solute_feats)
        solute_prime = torch.mm(interaction_map, solvent_feats)
        
        # Prediction phase
        solute_out = torch.cat([solute_feats, solute_prime], dim=1)
        solvent_out = torch.cat([solvent_feats, solvent_prime], dim=1)
        
        # Global pooling
        solute_pool = solute_out.mean(dim=0, keepdim=True)
        solvent_pool = solvent_out.mean(dim=0, keepdim=True)
        
        # Final prediction
        out = torch.cat([solute_pool, solvent_pool], dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        if return_attention:
            return out, interaction_map, self.solute_encoder.layers[-1].attention.attn_weights
        return out, interaction_map
