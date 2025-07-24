import numpy as np
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatherModel(nn.Module):
    def __init__(self, node_input_dim=42, edge_input_dim=10, 
                 node_hidden_dim=42, edge_hidden_dim=42, 
                 num_step_message_passing=6):
        super().__init__()
        
        # Enhanced edge processing
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_hidden_dim),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )
        
        self.conv = NNConv(
            in_feats=node_hidden_dim,
            out_feats=node_hidden_dim,
            edge_func=self.edge_network,
            aggregator_type='mean',  # Changed from sum to mean
            residual=True
        )
        
        # Hierarchical message passing
        self.num_steps = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        
        # Subgraph aggregation
        self.subgraph_proj = nn.Linear(node_hidden_dim * 2, node_hidden_dim)

    def forward(self, g, n_feat, e_feat):
        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        
        # First-level atomic aggregation
        for _ in range(self.num_steps // 2):
            m = torch.relu(self.conv(g, out, e_feat)) if e_feat is not None \
                else torch.relu(self.conv.bias + self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        
        # Second-level functional group aggregation
        with g.local_scope():
            g.ndata['h'] = out
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_group'))
            group_feat = g.ndata['h_group']
            out = self.subgraph_proj(torch.cat([out, group_feat], dim=1))
        
        return out + init

class EnhancedCIGINModel(nn.Module):
    def __init__(self, node_input_dim=42, edge_input_dim=10, 
                 node_hidden_dim=42, edge_hidden_dim=42,
                 num_step_message_passing=6, interaction='dot',
                 num_step_set2_set=2, num_layer_set2set=1):
        super().__init__()
        
        self.node_hidden_dim = node_hidden_dim
        self.interaction = interaction
        
        # Learnable interaction scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Enhanced gather models
        self.solute_gather = GatherModel(
            node_input_dim, edge_input_dim, 
            node_hidden_dim, edge_hidden_dim,
            num_step_message_passing
        )
        self.solvent_gather = GatherModel(
            node_input_dim, edge_input_dim,
            node_hidden_dim, edge_hidden_dim,
            num_step_message_passing
        )
        
        # Residual Set2Set pooling
        self.set2set_solute = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)
        
        # Multi-task prediction heads
        self.fc1 = nn.Linear(8 * node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)
        
        # Auxiliary prediction head
        self.aux_head = nn.Sequential(
            nn.Linear(8 * node_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predicts [logP, TPSA, QED]
        )

    def forward(self, data):
        solute, solvent, solute_len, solvent_len = data
        
        # Node embeddings
        solute_features = self.solute_gather(
            solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            solvent_features = self.solvent_gather(
                solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            solvent_features = self.solvent_gather(
                solvent, solvent.ndata['x'].float(), None)

        # Enhanced interaction phase
        len_map = torch.mm(solute_len.t(), solvent_len)
        
        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            Z = torch.cat([
                X1.repeat(solvent_features.shape[0], 1, 1),
                Y1.repeat(1, solute_features.shape[0], 1)
            ], -1)
            
            interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            
        else:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (self.temperature.abs() + 1e-8)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)
        
        ret_interaction_map = torch.clone(interaction_map)
        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Prediction phase with residual connections
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)
        
        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)
        
        final_features = torch.cat((solute_features, solvent_features), 1)
        
        # Main prediction
        main_pred = F.relu(self.fc1(final_features))
        main_pred = F.relu(self.fc2(main_pred))
        main_pred = self.fc3(main_pred)
        
        # Auxiliary predictions
        aux_pred = self.aux_head(final_features.detach())
        
        return main_pred, aux_pred, ret_interaction_map
