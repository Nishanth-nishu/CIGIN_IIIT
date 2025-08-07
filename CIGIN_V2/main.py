# python imports
import pandas as pd
import warnings
import os
import argparse
import numpy as np
import time

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
from train import train_full_data, get_metrics
from molecular_graph import get_graph_from_smile
from utils import *

# Disable RDKit and warning logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description='CIGIN Model Training on 100% Data')
parser.add_argument('--name', default='cigin_full', help="Project name (default: cigin_full)")
parser.add_argument('--interaction', default='dot', 
                    choices=['dot', 'scaled-dot', 'general', 'tanh-general'],
                    help="Interaction function type")
parser.add_argument('--max_epochs', type=int, default=100,
                    help="Maximum training epochs (default: 100)")
parser.add_argument('--batch_size', type=int, default=32,
                    help="Training batch size (default: 32)")
parser.add_argument('--model_type', default='both', 
                    choices=['original', 'transformer', 'both'],
                    help="Which model(s) to train")
parser.add_argument('--num_heads', type=int, default=6,
                    help="Number of attention heads (default: 6)")
parser.add_argument('--hidden_dim', type=int, default=42,
                    help="Hidden dimension (default: 42 to match original)")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate (default: 0.001)")
parser.add_argument('--scheduler_patience', type=int, default=5,
                    help="LR scheduler patience (default: 5)")

args = parser.parse_args()

# Configuration
config = {
    'project_name': args.name,
    'interaction': args.interaction,
    'max_epochs': args.max_epochs,
    'batch_size': args.batch_size,
    'model_type': args.model_type,
    'num_heads': args.num_heads,
    'hidden_dim': args.hidden_dim,
    'learning_rate': args.lr,
    'scheduler_patience': args.scheduler_patience
}

# Ensure hidden_dim is divisible by num_heads
if config['hidden_dim'] % config['num_heads'] != 0:
    config['hidden_dim'] = ((config['hidden_dim'] // config['num_heads']) + 1) * config['num_heads']
    print(f"Adjusted hidden_dim to {config['hidden_dim']} to be divisible by {config['num_heads']} heads")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
if not os.path.isdir(f"runs/run-{config['project_name']}"):
    os.makedirs(f"./runs/run-{config['project_name']}")
    os.makedirs(f"./runs/run-{config['project_name']}/models")

def collate(samples):
    """Batch preparation function (identical to original)"""
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes().tolist())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes().tolist())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels

class Dataclass(Dataset):
    """Custom dataset class (identical to original)"""
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

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_save_model(model, model_name, config, train_loader):
    """Train model on full dataset and save"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} on 100% data")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Config: {config}")
    print(f"{'='*50}")
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, 
        patience=config['scheduler_patience'], 
        mode='min', 
        verbose=True
    )
    
    start_time = time.time()
    
    # Train on full data
    train_full_data(
        config['max_epochs'],
        model,
        optimizer,
        scheduler,
        train_loader,
        f"{config['project_name']}_{model_name}"
    )
    
    training_time = time.time() - start_time
    
    print(f"\n{model_name} Training Complete:")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"Model saved to: ./runs/run-{config['project_name']}/models/{model_name}_final.tar")
    
    return {
        'model_name': model_name,
        'parameters': count_parameters(model),
        'training_time': training_time,
        'config': config
    }

def main():
    # Data loading - USE 100% OF DATA
    print("Loading data...")
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    
    print(f"Training on 100% of dataset: {len(df)} samples")

    # Create dataset and loader for ALL data
    full_dataset = Dataclass(df)
    train_loader = DataLoader(
        full_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate
    )
    
    results = []
    
    # Model training based on selection
    if config['model_type'] in ['original', 'both']:
        original_model = CIGINModel(
            interaction=config['interaction'],
            node_hidden_dim=42  # Force original dimension
        )
        results.append(train_and_save_model(
            original_model, 
            "Original_CIGIN", 
            config,
            train_loader
        ))
    
    if config['model_type'] in ['transformer', 'both']:
        transformer_model = CIGINGraphTransformerModel(
            interaction=config['interaction'],
            node_hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads']
        )
        results.append(train_and_save_model(
            transformer_model,
            "GraphTransformer_CIGIN",
            config,
            train_loader
        ))
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<25} {'Params':<12} {'Time (min)':<10}")
    print('-'*50)
    for r in results:
        print(f"{r['model_name']:<25} {r['parameters']:<12,} {r['training_time']/60:<10.2f}")
    
    # Save results
    pd.DataFrame(results).to_csv(
        f"runs/run-{config['project_name']}/training_results.csv",
        index=False
    )
    
    print(f"\nAll models trained on 100% data ({len(df)} samples)")
    print(f"Results saved to: runs/run-{config['project_name']}/training_results.csv")

if __name__ == '__main__':
    main()
