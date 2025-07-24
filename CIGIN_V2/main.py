# python imports
import pandas as pd
import warnings
import os
import argparse
from sklearn.model_selection import train_test_split
import numpy as np

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# dgl imports
import dgl

# local imports
from model import CIGINModel, CIGINGraphTransformerModel
from train import train, get_metrics
from molecular_graph import get_graph_from_smile
from utils import *

# Disable RDKit and warning logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description='CIGIN Model Training')
parser.add_argument('--name', default='cigin', help="The name of the current project")
parser.add_argument('--model_type', choices=['original', 'transformer'], default='original',
                    help="Type of model to use: original or transformer")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | tanh-general", 
                    default='dot')
parser.add_argument('--max_epochs', type=int, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', type=int, default=32, help="The batch size for training")
parser.add_argument('--num_layers', type=int, default=6, help="Number of message passing/transformer layers")
parser.add_argument('--num_heads', type=int, default=6, help="Number of attention heads (for transformer only)")
parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate (for transformer only)")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")

args = parser.parse_args()
project_name = args.name
model_type = args.model_type
interaction = args.interaction
max_epochs = args.max_epochs
batch_size = args.batch_size
num_layers = args.num_layers
num_heads = args.num_heads
dropout = args.dropout
learning_rate = args.learning_rate

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Create output directories
os.makedirs(f"runs/run-{project_name}/models", exist_ok=True)

def collate(samples):
    """Batch preparation function"""
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes().tolist())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes().tolist())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, torch.tensor(labels, dtype=torch.float32)

class SolvationDataset(Dataset):
    """Custom dataset class for solvation data"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Process solute
        solute = self.dataset.iloc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)

        # Process solvent
        solvent = self.dataset.iloc[idx]['SolventSMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent)

        delta_g = self.dataset.iloc[idx]['delGsolv']
        return solute_graph, solvent_graph, delta_g

def initialize_model():
    """Initialize the appropriate model based on arguments"""
    if model_type == 'transformer':
        print(f"Initializing Graph Transformer model with {num_layers} layers and {num_heads} heads")
        model = CIGINGraphTransformerModel(
            interaction=interaction,
            num_step_message_passing=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    else:
        print(f"Initializing original CIGIN model with {num_layers} message passing layers")
        model = CIGINModel(
            interaction=interaction,
            num_step_message_passing=num_layers
        )
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    
    # Split data into train/valid/test (80/10/10)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.111, random_state=42)

    # Create datasets and data loaders
    print("Creating datasets and data loaders...")
    train_loader = DataLoader(
        SolvationDataset(train_df), 
        collate_fn=collate, 
        batch_size=batch_size, 
        shuffle=True
    )
    valid_loader = DataLoader(
        SolvationDataset(valid_df), 
        collate_fn=collate, 
        batch_size=128
    )
    test_loader = DataLoader(
        SolvationDataset(test_df), 
        collate_fn=collate, 
        batch_size=128
    )

    # Initialize model
    model = initialize_model()
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    # Training loop
    print(f"\nStarting training for {max_epochs} epochs...")
    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)

    # Final evaluation on test set
    model.eval()
    loss, mae_loss = get_metrics(model, test_loader)
    print(f"\nFinal test set performance:")
    print(f"MSE Loss: {loss:.4f}")
    print(f"MAE Loss: {mae_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f"runs/run-{project_name}/models/final_model.pt")
    print(f"Model saved to runs/run-{project_name}/models/final_model.pt")

if __name__ == '__main__':
    main()
