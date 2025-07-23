import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphTransformerLayer
import math

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder for molecular graphs
    Replaces the GatherModel (MPNN) in the original CIGIN
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 num_heads=4,
                 num_layers=6,
                 dropout=0.1,
                 use_edge_features=True):
        super(GraphTransformerEncoder, self).__init__()
        
        self.node_hidden_dim = node_hidden_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        
        # Input projection
        self.node_projection = nn.Linear(node_input_dim, node_hidden_dim)
        
        if use_edge_features:
            self.edge_projection = nn.Linear(edge_input_dim, node_hidden_dim)
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_feat=node_hidden_dim,
                out_feat=node_hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=True,
                batch_norm=False,
                residual=True,
                use_bias=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_hidden_dim)
        
    def forward(self, g, node_feat, edge_feat=None):
        """
        Forward pass through Graph Transformer
        
        Parameters:
        -----------
        g : DGLGraph
            Input molecular graph
        node_feat : torch.Tensor
            Node features [num_nodes, node_input_dim]
        edge_feat : torch.Tensor, optional
            Edge features [num_edges, edge_input_dim]
            
        Returns:
        --------
        torch.Tensor
            Node embeddings [num_nodes, node_hidden_dim]
        """
        # Project input features
        h = self.node_projection(node_feat)
        
        # Handle edge features if available
        if self.use_edge_features and edge_feat is not None:
            e = self.edge_projection(edge_feat)
        else:
            e = None
            
        # Apply transformer layers
        for layer in self.transformer_layers:
            h = layer(g, h, e)
            
        # Final layer normalization
        h = self.layer_norm(h)
        
        return h

class GraphTransformerPooling(nn.Module):
    """
    Graph-level pooling using transformer-style attention
    Replaces Set2Set pooling
    """
    def __init__(self, node_dim, hidden_dim=None):
        super(GraphTransformerPooling, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = node_dim
            
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=False
        )
        
        # Learnable query vector for pooling
        self.query = nn.Parameter(torch.randn(1, node_dim))
        
        # Output projection
        self.output_proj = nn.Linear(node_dim, hidden_dim)
        
    def forward(self, g, node_embeddings):
        """
        Pool node embeddings to graph-level representation
        
        Parameters:
        -----------
        g : DGLGraph
            Batched molecular graphs
        node_embeddings : torch.Tensor
            Node embeddings [total_nodes, node_dim]
            
        Returns:
        --------
        torch.Tensor
            Graph-level embeddings [batch_size, hidden_dim]
        """
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
        
        graph_embeddings = []
        start_idx = 0
        
        for i, num_nodes in enumerate(batch_num_nodes):
            if num_nodes == 0:
                # Handle empty graphs
                graph_embeddings.append(torch.zeros(1, self.hidden_dim, device=node_embeddings.device))
                continue
                
            # Extract nodes for current graph
            end_idx = start_idx + num_nodes
            graph_nodes = node_embeddings[start_idx:end_idx]  # [num_nodes, node_dim]
            
            # Prepare for attention: [seq_len, batch, embed_dim]
            nodes_for_attention = graph_nodes.unsqueeze(1).transpose(0, 1)  # [num_nodes, 1, node_dim]
            query_for_attention = self.query.unsqueeze(1)  # [1, 1, node_dim]
            
            # Apply attention pooling
            pooled, _ = self.attention(
                query_for_attention,
                nodes_for_attention,
                nodes_for_attention
            )
            
            # Project to output dimension
            pooled = self.output_proj(pooled.squeeze(1))  # [1, hidden_dim]
            graph_embeddings.append(pooled)
            
            start_idx = end_idx
            
        return torch.cat(graph_embeddings, dim=0)  # [batch_size, hidden_dim]

class CIGINModel(nn.Module):
    """
    CIGIN model with Graph Transformer encoders
    Minimal changes from original - only replaced GatherModel and Set2Set
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_transformer_layers=6,
                 num_heads=4,
                 interaction='dot',
                 dropout=0.1):
        super(CIGINModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.interaction = interaction
        
        # Replace GatherModel with GraphTransformerEncoder
        self.solute_gather = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            node_hidden_dim=node_hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.solvent_gather = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            node_hidden_dim=node_hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Replace Set2Set with GraphTransformerPooling
        # Note: output dimension is 2 * node_hidden_dim to match original Set2Set output
        self.pooling_solute = GraphTransformerPooling(
            node_dim=2 * node_hidden_dim,
            hidden_dim=4 * node_hidden_dim  # Match original Set2Set output size
        )
        
        self.pooling_solvent = GraphTransformerPooling(
            node_dim=2 * node_hidden_dim,
            hidden_dim=4 * node_hidden_dim  # Match original Set2Set output size
        )
        
        # Keep original prediction layers unchanged
        self.fc1 = nn.Linear(8 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        
        # Get node embeddings using Graph Transformers (replaces gather models)
        solute_features = self.solute_gather(
            solute, 
            solute.ndata['x'].float(), 
            solute.edata['w'].float() if 'w' in solute.edata else None
        )
        
        try:
            solvent_features = self.solvent_gather(
                solvent, 
                solvent.ndata['x'].float(), 
                solvent.edata['w'].float() if 'w' in solvent.edata else None
            )
        except:
            solvent_features = self.solvent_gather(
                solvent, 
                solvent.ndata['x'].float(), 
                None
            )
        
        # Interaction phase (unchanged from original)
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

        # Cross-attention features (unchanged from original)
        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Concatenate original and cross-attended features
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        # Graph-level pooling using transformer pooling (replaces Set2Set)
        solute_features = self.pooling_solute(solute, solute_features)
        solvent_features = self.pooling_solvent(solvent, solvent_features)

        # Final prediction (unchanged from original)
        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map
