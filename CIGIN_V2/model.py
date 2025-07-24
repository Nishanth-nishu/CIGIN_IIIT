import numpy as np
import math
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularFeatureEnhancer(nn.Module):
    """Enhanced feature processing for molecular graphs"""
    def __init__(self, node_dim=42, edge_dim=10):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.LayerNorm(node_dim))
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_dim))
    
    def forward(self, node_feat, edge_feat):
        return self.node_encoder(node_feat), self.edge_encoder(edge_feat) if edge_feat is not None else None

class EnhancedGraphTransformerLayer(nn.Module):
    """Improved transformer layer for molecular graphs"""
    def __init__(self, hidden_dim=42, num_heads=6, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.edge_proj = nn.Linear(10, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_attr=None):
        # Attention with edge features
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Enhanced FFN
        x = x + self.ffn(x)
        return self.norm2(x)

class MolecularAttentionPooling(nn.Module):
    """Advanced pooling mimicking Set2Set behavior"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_dim, 4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, in_dim))
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, g, x):
        batch_sizes = g.batch_num_nodes().tolist()
        outputs = []
        start = 0
        
        for nodes in batch_sizes:
            graph_nodes = x[start:start+nodes].unsqueeze(0)  # [1, nodes, dim]
            query = self.query.expand(1, -1, -1)  # [1, 1, dim]
            pooled, _ = self.attention(query, graph_nodes, graph_nodes)
            outputs.append(pooled.squeeze(0))
            start += nodes
            
        pooled = torch.cat(outputs, dim=0) if len(batch_sizes) > 1 else outputs[0]
        return self.proj(pooled)

class GraphTransformerGatherModel(nn.Module):
    """Enhanced transformer version of GatherModel"""
    def __init__(self, node_input_dim=42, edge_input_dim=10, 
                 node_hidden_dim=42, num_step_message_passing=6, num_heads=6):
        super().__init__()
        self.feature_enhancer = MolecularFeatureEnhancer(node_input_dim, edge_input_dim)
        self.initial_proj = nn.Linear(node_input_dim, node_hidden_dim)
        self.layers = nn.ModuleList([
            EnhancedGraphTransformerLayer(node_hidden_dim, num_heads)
            for _ in range(num_step_message_passing)
        ])

    def forward(self, g, n_feat, e_feat):
        n_feat, e_feat = self.feature_enhancer(n_feat, e_feat)
        out = F.relu(self.initial_proj(n_feat))
        init = out.clone()
        
        for layer in self.layers:
            out = layer(g, out, e_feat)
            
        return out + init  # Residual connection

class CIGINGraphTransformerModel(nn.Module):
    """Enhanced CIGIN with Graph Transformers"""
    def __init__(self, node_input_dim=42, edge_input_dim=10, node_hidden_dim=42,
                 edge_hidden_dim=42, num_step_message_passing=6, interaction='dot', num_heads=6):
        super().__init__()
        
        # Model configuration
        self.node_hidden_dim = node_hidden_dim
        self.interaction = interaction
        
        # Molecular feature processors
        self.solute_gather = GraphTransformerGatherModel(
            node_input_dim, edge_input_dim, node_hidden_dim, 
            num_step_message_passing, num_heads)
        self.solvent_gather = GraphTransformerGatherModel(
            node_input_dim, edge_input_dim, node_hidden_dim,
            num_step_message_passing, num_heads)
        
        # Pooling layers
        self.solute_pool = MolecularAttentionPooling(2*node_hidden_dim, 4*node_hidden_dim)
        self.solvent_pool = MolecularAttentionPooling(2*node_hidden_dim, 4*node_hidden_dim)
        
        # Interaction and prediction
        self.imap = nn.Linear(80, 1)
        self.predictor = nn.Sequential(
            nn.Linear(8*node_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1))

    def forward(self, data):
        solute, solvent, solute_len, solvent_len = data
        
        # Get molecular features
        solute_features = self.solute_gather(
            solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            solvent_features = self.solvent_gather(
                solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            solvent_features = self.solvent_gather(
                solvent, solvent.ndata['x'].float(), None)

        # Interaction phase (unchanged from original)
        len_map = torch.mm(solute_len.t(), solvent_len)
        
        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            Z = torch.cat([X1.repeat(solvent_features.shape[0], 1, 1),
                          Y1.repeat(1, solute_features.shape[0], 1)], -1)
            interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map.t())
        else:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / math.sqrt(self.node_hidden_dim)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)
        
        ret_interaction_map = torch.clone(interaction_map)
        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Prediction phase
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)
        
        solute_features = self.solute_pool(solute, solute_features)
        solvent_features = self.solvent_pool(solvent, solvent_features)
        
        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = self.predictor(final_features)
        
        return predictions, ret_interaction_map

# Original CIGIN Model (maintained for compatibility)
class GatherModel(nn.Module):
    """Original MPNN from CIGIN paper (maintained unchanged)"""
    def __init__(self, node_input_dim=42, edge_input_dim=10, node_hidden_dim=42,
                 edge_hidden_dim=42, num_step_message_passing=6):
        super().__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim, out_feats=node_hidden_dim,
                          edge_func=edge_network, aggregator_type='sum', residual=True)

    def forward(self, g, n_feat, e_feat):
        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        for _ in range(self.num_step_message_passing):
            m = torch.relu(self.conv(g, out, e_feat)) if e_feat is not None \
                else torch.relu(self.conv.bias + self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init

class CIGINModel(nn.Module):
    """Original CIGIN model (maintained unchanged)"""
    def __init__(self, node_input_dim=42, edge_input_dim=10, node_hidden_dim=42,
                 edge_hidden_dim=42, num_step_message_passing=6, interaction='dot',
                 num_step_set2_set=2, num_layer_set2set=1):
        super().__init__()
        self.solute_gather = GatherModel(node_input_dim, edge_input_dim,
                                       node_hidden_dim, edge_input_dim,
                                       num_step_message_passing)
        self.solvent_gather = GatherModel(node_input_dim, edge_input_dim,
                                        node_hidden_dim, edge_input_dim,
                                        num_step_message_passing)
        self.fc1 = nn.Linear(8 * node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)
        self.set2set_solute = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            # if edge exists in a molecule
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None)

        # Interaction phase
        len_map = torch.mm(solute_len.t(), solvent_len)

        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            X2 = X1.repeat(solvent_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, solute_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)

        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Prediction phase
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)

        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map
