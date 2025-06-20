import numpy as np
from dgl.nn.pytorch import Set2Set, NNConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerAggregator(nn.Module):
    """
    Simple Transformer-based aggregation to replace Set2Set
    """
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerAggregator, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Ensure transformer_dim is divisible by num_heads
        self.transformer_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads
        
        # Input projection to transformer dimension
        self.input_proj = nn.Linear(input_dim, self.transformer_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=num_heads,
            dim_feedforward=self.transformer_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to match Set2Set output (2 * input_dim)
        self.output_proj = nn.Linear(self.transformer_dim, 2 * input_dim)
        
        # Learnable global token for aggregation
        self.global_token = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
        
    def forward(self, graph, node_features):
        """
        Forward pass with batched graphs
        Args:
            graph: DGL batched graph
            node_features: [total_nodes, input_dim]
        Returns:
            aggregated_features: [batch_size, 2 * input_dim]
        """
        try:
            device = node_features.device
            batch_sizes = graph.batch_num_nodes().cpu().numpy()
            batch_size = len(batch_sizes)
            
            # Project input features to transformer dimension
            node_features_proj = self.input_proj(node_features)
            
            # Split node features by batch
            aggregated_outputs = []
            start_idx = 0
            
            for i in range(batch_size):
                end_idx = start_idx + batch_sizes[i]
                batch_nodes = node_features_proj[start_idx:end_idx]  # [num_nodes_i, transformer_dim]
                
                # Add global token
                global_token = self.global_token.expand(1, -1, -1).to(device)  # [1, 1, transformer_dim]
                
                # Combine global token with node features
                # [1 + num_nodes_i, transformer_dim]
                combined = torch.cat([global_token.squeeze(0), batch_nodes], dim=0)
                combined = combined.unsqueeze(0)  # [1, 1 + num_nodes_i, transformer_dim]
                
                # Apply transformer
                transformed = self.transformer(combined)  # [1, 1 + num_nodes_i, transformer_dim]
                
                # Take the global token (first token) as the graph representation
                graph_repr = transformed[:, 0, :]  # [1, transformer_dim]
                
                # Project to output dimension
                output = self.output_proj(graph_repr)  # [1, 2 * input_dim]
                aggregated_outputs.append(output)
                
                start_idx = end_idx
            
            # Concatenate all batch outputs
            final_output = torch.cat(aggregated_outputs, dim=0)  # [batch_size, 2 * input_dim]
            
            return final_output
            
        except Exception as e:
            print(f"Error in TransformerAggregator: {e}")
            # Fallback to simple mean pooling
            batch_sizes = graph.batch_num_nodes().cpu().numpy()
            batch_size = len(batch_sizes)
            
            batch_outputs = []
            start_idx = 0
            for i in range(batch_size):
                end_idx = start_idx + batch_sizes[i]
                batch_nodes = node_features[start_idx:end_idx]
                
                # Simple mean pooling
                mean_pooled = torch.mean(batch_nodes, dim=0, keepdim=True)
                # Duplicate to match Set2Set output dimension
                output = torch.cat([mean_pooled, mean_pooled], dim=1)
                batch_outputs.append(output)
                start_idx = end_idx
            
            return torch.cat(batch_outputs, dim=0)


class GatherModel(nn.Module):
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True)

    def forward(self, g, n_feat, e_feat):
        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g, out, e_feat))
            else:
                m = torch.relu(self.conv.bias + self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init


class CIGINModel(nn.Module):
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 use_transformer=False,
                 transformer_heads=4,
                 transformer_layers=2):
        super(CIGINModel, self).__init__()

        self.node_hidden_dim = node_hidden_dim
        self.interaction = interaction
        self.use_transformer = use_transformer

        self.solute_gather = GatherModel(node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, num_step_message_passing)
        self.solvent_gather = GatherModel(node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, num_step_message_passing)

        # Choose aggregation method
        if use_transformer:
            print("🤖 Using Transformer aggregation")
            # Input dimension is 2 * node_hidden_dim (after interaction concatenation)
            self.set2set_solute = TransformerAggregator(
                2 * node_hidden_dim, 
                num_heads=transformer_heads, 
                num_layers=transformer_layers
            )
            self.set2set_solvent = TransformerAggregator(
                2 * node_hidden_dim, 
                num_heads=transformer_heads, 
                num_layers=transformer_layers
            )
        else:
            print("📊 Using Set2Set aggregation")
            self.set2set_solute = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)
            self.set2set_solvent = Set2Set(2 * node_hidden_dim, num_step_set2_set, num_layer_set2set)

        self.fc1 = nn.Linear(8 * node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.imap = nn.Linear(80, 1)

    def forward(self, data):
        solute, solvent, solute_len, solvent_len = data

        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None)

        len_map = torch.mm(solute_len.t(), solvent_len)

        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            X2 = X1.repeat(solvent_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, solute_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            elif self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)
            else:
                raise ValueError("Invalid interaction type")

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        else:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)

        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)

        final_features = torch.cat((solute_features, solvent_features), dim=1)

        predictions = F.relu(self.fc1(final_features))
        predictions = F.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map