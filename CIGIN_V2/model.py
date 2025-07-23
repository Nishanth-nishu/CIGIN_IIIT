import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphTransformerLayer
from math import sqrt

class GraphTransformerGather(nn.Module):
    def __init__(self, node_dim=42, edge_dim=10, num_heads=4, num_layers=3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Graph Transformer layers with residual connections
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                in_size=node_dim,
                out_size=node_dim,
                num_heads=num_heads,
                dropout=0.1,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        
    def forward(self, g, node_feats, edge_feats=None):
        # Compute attention biases from edge features
        attn_bias = torch.zeros(g.num_nodes(), g.num_nodes(), device=node_feats.device)
        if edge_feats is not None:
            edge_bias = self.edge_proj(edge_feats)  # [E, num_heads]
            g.edata['a'] = edge_bias
            g.update_all(fn.copy_e('a', 'm'), fn.sum('m', 'a_sum'))
            attn_bias = g.ndata['a_sum'].permute(1, 0)  # [N, N, H]
        
        # Transformer processing
        h = node_feats
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(g, h, attn_bias=attn_bias))
        return h

class CIGINModel(nn.Module):
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
        
        if return_attention:
            # For attention visualization (requires modification to GraphTransformerLayer)
            return out, interaction_map, None  # Replace with actual attention weights
        return out, interaction_map
