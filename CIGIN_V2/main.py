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
from model import CIGINModel
from train import train, get_metrics
from molecular_graph import get_graph_from_smile
from utils import *

# Disable RDKit and warning logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin_transformer', help="The name of the current project: default: CIGIN_Transformer")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | tanh-general", 
                    default='dot')
parser.add_argument('--max_epochs', required=False, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=32, help="The batch size for training")
parser.add_argument('--num_transformer_layers', required=False, default=6, help="Number of transformer layers")
parser.add_argument('--num_heads', required=False, default=4, help="Number of attention heads")
parser.add_argument('--dropout', required=False, default=0.1, help="Dropout rate")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)
num_transformer_layers = int(args.num_transformer_layers)
num_heads = int(args.num_heads)
dropout = float(args.dropout)

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Create output directories
if not os.path.isdir("runs/run-" + str(project_name)):
    os.makedirs("./runs/run-" + str(project_name))
    os.makedirs("./runs/run-" + str(project_name) + "/models")

def collate(samples):
    """Batch preparation function - unchanged from original"""
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes().tolist())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes().tolist())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels

class Dataclass(Dataset):
    """Custom dataset class - unchanged from original"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        solute = self.dataset.iloc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)

        solvent = self.dataset.iloc[idx]['SolventSMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent)

        delta_g = self.dataset.iloc[idx]['delGsolv']
        return [solute_graph, solvent_graph, [delta_g]]

def main():
    # Data loading and splitting - unchanged from original
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    
    # Split data into train/valid/test (80/10/10)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.111, random_state=42)

    # Create datasets
    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)
    test_dataset = Dataclass(test_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)
    test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=128)

    # Initialize model with Graph Transformer
    model = CIGINModel(
        interaction=interaction,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    model.to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using Graph Transformer with {num_transformer_layers} layers and {num_heads} heads")
    
    # Training setup - unchanged from original
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    # Training loop - unchanged from original
    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)

    # Final evaluation on test set - unchanged from original
    model.eval()
    loss, mae_loss = get_metrics(model, test_loader)
    print(f"\nFinal test set performance:")
    print(f"MSE Loss: {loss:.4f}")
    print(f"MAE Loss: {mae_loss:.4f}")

if __name__ == '__main__':
    main()
