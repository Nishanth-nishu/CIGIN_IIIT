import numpy as np
import math

from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer that replaces message passing
    """
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Adjust num_heads to be compatible with out_dim
        if out_dim % num_heads != 0:
            # Find the largest divisor of out_dim that's <= num_heads
            for h in range(min(num_heads, out_dim), 0, -1):
                if out_dim % h == 0:
                    num_heads = h
                    break
        
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Edge embedding for positional encoding (handle edge_input_dim properly)
        self.edge_encoder = nn.Linear(10, num_heads) if num_heads > 1 else nn.Linear(10, 1)
        
        print(f"GraphTransformerLayer: in_dim={in_dim}, out_dim={out_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
    def forward(self, g, node_feat, edge_feat=None):
        batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
        num_nodes = node_feat.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(node_feat).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(node_feat).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(node_feat).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('ihd,jhd->ijh', Q, K) / math.sqrt(self.head_dim)
        
        # Add edge features as bias if available
        if edge_feat is not None:
            # Get edge indices
            src, dst = g.edges()
            edge_bias = self.edge_encoder(edge_feat)
            if self.num_heads > 1:
                edge_bias = edge_bias.view(-1, self.num_heads)
            else:
                edge_bias = edge_bias.view(-1, 1).expand(-1, self.num_heads)
            
            # Create attention bias matrix
            bias_matrix = torch.zeros(num_nodes, num_nodes, self.num_heads, device=scores.device)
            bias_matrix[src, dst] = edge_bias
            scores = scores + bias_matrix
        
        # Create adjacency mask (only attend to connected nodes)
        adj_mask = torch.zeros(num_nodes, num_nodes, device=scores.device)
        if g.number_of_edges() > 0:
            src, dst = g.edges()
            adj_mask[src, dst] = 1
            adj_mask[dst, src] = 1  # Make it symmetric for undirected graphs
        
        # Add self-connections
        adj_mask.fill_diagonal_(1)
        
        # Apply mask
        scores = scores.masked_fill(adj_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.einsum('ijh,jhd->ihd', attn_weights, V)
        out = out.contiguous().view(num_nodes, -1)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm1(out + node_feat if out.size(-1) == node_feat.size(-1) else out)
        
        # Feed-forward network
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out


class GraphTransformerPooling(nn.Module):
    """
    Graph Transformer-based pooling that replaces Set2Set
    """
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(GraphTransformerPooling, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Adjust num_heads to be compatible with in_dim
        if in_dim % num_heads != 0:
            for h in range(min(num_heads, in_dim), 0, -1):
                if in_dim % h == 0:
                    num_heads = h
                    break
        
        self.num_heads = num_heads
        print(f"GraphTransformerPooling: in_dim={in_dim}, out_dim={out_dim}, num_heads={num_heads}")
        
        # Global attention for pooling
        self.global_attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Learnable query vector for global pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, in_dim))
        
        # Output projection to match Set2Set output dimension
        self.output_proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, g, node_feat):
        # Get batch information
        if hasattr(g, 'batch_num_nodes'):
            batch_sizes = g.batch_num_nodes().tolist()
            batch_size = len(batch_sizes)
        else:
            batch_sizes = [node_feat.size(0)]
            batch_size = 1
        
        outputs = []
        start_idx = 0
        
        for i, num_nodes in enumerate(batch_sizes):
            # Extract nodes for current graph
            graph_nodes = node_feat[start_idx:start_idx + num_nodes].unsqueeze(0)  # [1, num_nodes, feat_dim]
            
            # Global attention pooling
            global_query = self.global_query.expand(1, -1, -1)  # [1, 1, feat_dim]
            
            pooled_feat, _ = self.global_attention(
                global_query, graph_nodes, graph_nodes
            )  # [1, 1, feat_dim]
            
            pooled_feat = pooled_feat.squeeze(0).squeeze(0)  # [feat_dim]
            outputs.append(pooled_feat)
            start_idx += num_nodes
        
        # Stack outputs
        if batch_size > 1:
            output = torch.stack(outputs, dim=0)  # [batch_size, feat_dim]
        else:
            output = outputs[0].unsqueeze(0)  # [1, feat_dim]
        
        # Project to desired output dimension
        output = self.output_proj(output)
        
        return output


class GraphTransformerGatherModel(nn.Module):
    """
    Graph Transformer version of GatherModel
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 num_heads=8):
        super(GraphTransformerGatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.node_hidden_dim = node_hidden_dim
        
        # Input projection
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        
        # Adjust num_heads to be compatible with node_hidden_dim
        if node_hidden_dim % num_heads != 0:
            for h in range(min(num_heads, node_hidden_dim), 0, -1):
                if node_hidden_dim % h == 0:
                    num_heads = h
                    break
        
        print(f"GraphTransformerGatherModel: Using {num_heads} heads for hidden_dim={node_hidden_dim}")
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=node_hidden_dim,
                out_dim=node_hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_step_message_passing)
        ])
        
    def forward(self, g, n_feat, e_feat):
        """Returns the node embeddings after graph transformer layers.
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : node features
        """
        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            out = transformer_layer(g, out, e_feat)
            
        return out + init


class CIGINGraphTransformerModel(nn.Module):
    """
    CIGIN model with Graph Transformer layers replacing Set2Set and message passing
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_heads=8):
        super(CIGINGraphTransformerModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        
        # Graph Transformer gather models
        self.solute_gather = GraphTransformerGatherModel(
            self.node_input_dim, self.edge_input_dim,
            self.node_hidden_dim, self.edge_input_dim,
            self.num_step_message_passing, num_heads
        )
        self.solvent_gather = GraphTransformerGatherModel(
            self.node_input_dim, self.edge_input_dim,
            self.node_hidden_dim, self.edge_input_dim,
            self.num_step_message_passing, num_heads
        )
        
        # Graph Transformer pooling layers (replacing Set2Set)
        self.transformer_pool_solute = GraphTransformerPooling(
            in_dim=2 * node_hidden_dim,
            out_dim=4 * node_hidden_dim,  # Match Set2Set output dimension
            num_heads=4
        )
        self.transformer_pool_solvent = GraphTransformerPooling(
            in_dim=2 * node_hidden_dim,
            out_dim=4 * node_hidden_dim,  # Match Set2Set output dimension
            num_heads=4
        )
        
        # These three are the FFNN for prediction phase (unchanged)
        self.fc1 = nn.Linear(8 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        
        # Node embeddings after graph transformer phase
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            # If edge exists in a molecule
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            # If edge doesn't exist in a molecule, for example in case of water
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None)

        # Interaction phase (unchanged)
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

        # Prediction phase with Graph Transformer pooling
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        # Use Graph Transformer pooling instead of Set2Set
        solute_features = self.transformer_pool_solute(solute, solute_features)
        solvent_features = self.transformer_pool_solvent(solvent, solvent_features)

        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map


# Keep original models for comparison
class GatherModel(nn.Module):
    """
    Original MPNN from CIGIN paper (unchanged)
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True
                           )

    def forward(self, g, n_feat, e_feat):
        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g, out, e_feat))
            else:
                m = torch.relu(self.conv.bias +  self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init


class CIGINModel(nn.Module):
    """
    Original CIGIN model (unchanged)
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 ):
        super(CIGINModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )
        # These three are the FFNN for prediction phase
        self.fc1 = nn.Linear(8 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

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
