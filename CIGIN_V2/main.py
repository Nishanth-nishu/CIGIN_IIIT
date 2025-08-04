# python imports
import pandas as pd
import warnings
import os
import argparse
from sklearn.model_selection import KFold
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
from train import train_cv, get_metrics
from molecular_graph import get_graph_from_smile
from utils import *

# Disable RDKit and warning logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description='CIGIN Model 10-fold CV with 5 repetitions')
parser.add_argument('--name', default='cigin_cv', help="Project name (default: cigin_cv)")
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
    'scheduler_patience': args.scheduler_patience,
    'valid_batch_size': 128,
    'test_batch_size': 128
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

def run_single_fold(model_class, train_data, val_data, config, fold_id, rep_id):
    """Run a single fold of cross-validation"""
    
    # Create fresh model instance
    if model_class == CIGINModel:
        model = CIGINModel(
            interaction=config['interaction'],
            node_hidden_dim=42
        )
    else:
        model = CIGINGraphTransformerModel(
            interaction=config['interaction'],
            node_hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads']
        )
    
    model.to(device)
    
    # Create data loaders
    train_dataset = Dataclass(train_data)
    val_dataset = Dataclass(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['valid_batch_size'],
        collate_fn=collate
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, 
        patience=config['scheduler_patience'], 
        mode='min', 
        verbose=False
    )
    
    # Train model
    train_cv(
        config['max_epochs'],
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader
    )
    
    # Evaluate on validation set
    val_loss, val_mae = get_metrics(model, val_loader)
    val_rmse = np.sqrt(val_loss)
    
    return val_rmse

def cross_validate_model(model_class, model_name, df, config):
    """Perform 10-fold cross-validation with 5 repetitions"""
    print(f"\n{'='*60}")
    print(f"Running 10-fold CV with 5 repetitions for {model_name}")
    print(f"{'='*60}")
    
    all_rmse_scores = []
    
    for rep in range(5):  # 5 repetitions
        print(f"\nRepetition {rep + 1}/5")
        rep_rmse_scores = []
        
        # Create KFold with different random state for each repetition
        kfold = KFold(n_splits=10, shuffle=True, random_state=42 + rep)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
            print(f"  Fold {fold + 1}/10", end=" ")
            
            train_data = df.iloc[train_idx].reset_index(drop=True)
            val_data = df.iloc[val_idx].reset_index(drop=True)
            
            rmse = run_single_fold(model_class, train_data, val_data, config, fold, rep)
            rep_rmse_scores.append(rmse)
            
            print(f"RMSE: {rmse:.4f}")
        
        rep_avg_rmse = np.mean(rep_rmse_scores)
        all_rmse_scores.extend(rep_rmse_scores)
        print(f"  Repetition {rep + 1} Average RMSE: {rep_avg_rmse:.4f}")
    
    # Calculate final statistics
    final_avg_rmse = np.mean(all_rmse_scores)
    final_std_rmse = np.std(all_rmse_scores)
    
    print(f"\n{model_name} Final Results:")
    print(f"Average RMSE over 50 folds (10×5): {final_avg_rmse:.4f} ± {final_std_rmse:.4f}")
    
    return {
        'model_name': model_name,
        'avg_rmse': final_avg_rmse,
        'std_rmse': final_std_rmse,
        'all_scores': all_rmse_scores
    }

def main():
    # Data loading
    print("Loading data...")
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    
    print(f"Total dataset size: {len(df)}")
    print(f"Each fold will have ~{len(df)//10} samples for validation")
    
    results = []
    
    # Run cross-validation for selected models
    if config['model_type'] in ['original', 'both']:
        results.append(cross_validate_model(
            CIGINModel, 
            "Original_CIGIN", 
            df, 
            config
        ))
    
    if config['model_type'] in ['transformer', 'both']:
        results.append(cross_validate_model(
            CIGINGraphTransformerModel,
            "GraphTransformer_CIGIN",
            df,
            config
        ))
    
    # Save and display final results
    print(f"\n{'='*80}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"{result['model_name']}: {result['avg_rmse']:.4f} ± {result['std_rmse']:.4f} RMSE")
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            'model': r['model_name'],
            'avg_rmse': r['avg_rmse'],
            'std_rmse': r['std_rmse'],
            'config': str(config)
        } for r in results
    ])
    
    results_df.to_csv(
        f"runs/run-{config['project_name']}/cv_results.csv",
        index=False
    )
    
    # Save all individual scores for further analysis
    for result in results:
        scores_df = pd.DataFrame({
            'fold_rmse': result['all_scores']
        })
        scores_df.to_csv(
            f"runs/run-{config['project_name']}/{result['model_name']}_all_scores.csv",
            index=False
        )

if __name__ == '__main__':
    main()
