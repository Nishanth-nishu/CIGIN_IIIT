# python imports
import pandas as pd
import warnings
import os
import argparse
from sklearn.model_selection import train_test_split
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
from model_graph_transformer import CIGINModel, CIGINGraphTransformerModel
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
parser.add_argument('--name', default='cigin', help="The name of the current project: default: CIGIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | tanh-general", 
                    default='dot')
parser.add_argument('--max_epochs', required=False, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=32, help="The batch size for training")
parser.add_argument('--model_type', required=False, default='both', choices=['original', 'transformer', 'both'],
                    help="Which model to train: original, transformer, or both")
parser.add_argument('--num_heads', required=False, default=8, help="Number of attention heads for transformer")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)
model_type = args.model_type
num_heads = int(args.num_heads)

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Create output directories
if not os.path.isdir("runs/run-" + str(project_name)):
    os.makedirs("./runs/run-" + str(project_name))
    os.makedirs("./runs/run-" + str(project_name) + "/models")

def collate(samples):
    """Batch preparation function"""
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes().tolist())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes().tolist())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels

class Dataclass(Dataset):
    """Custom dataset class"""
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
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_model(model, model_name, train_loader, valid_loader, test_loader, max_epochs):
    """Train and evaluate a single model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"{'='*50}")
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)
    
    # Record training time
    start_time = time.time()
    
    # Training
    project_name_model = f"{project_name}_{model_name}"
    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name_model)
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    model.eval()
    test_loss, test_mae = get_metrics(model, test_loader)
    
    print(f"\n{model_name} Results:")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Test MSE Loss: {test_loss:.4f}")
    print(f"Test MAE Loss: {test_mae:.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
    
    return {
        'model_name': model_name,
        'test_mse': test_loss,
        'test_mae': test_mae,
        'test_rmse': np.sqrt(test_loss),
        'training_time': training_time,
        'parameters': count_parameters(model)
    }

def main():
    # Data loading and splitting
    print("Loading and preparing data...")
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    
    print(f"Dataset size: {len(df)} samples")
    
    # Split data into train/valid/test (80/10/10)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.111, random_state=42)
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Create datasets
    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)
    test_dataset = Dataclass(test_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)
    test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=128)
    
    results = []
    
    # Train models based on user choice
    if model_type in ['original', 'both']:
        print("\nInitializing Original CIGIN Model (Set2Set + Message Passing)...")
        original_model = CIGINModel(interaction=interaction)
        results.append(train_and_evaluate_model(
            original_model, "Original_CIGIN", train_loader, valid_loader, test_loader, max_epochs
        ))
    
    if model_type in ['transformer', 'both']:
        print(f"\nInitializing Graph Transformer CIGIN Model (num_heads={num_heads})...")
        transformer_model = CIGINGraphTransformerModel(
            interaction=interaction,
            num_heads=num_heads
        )
        results.append(train_and_evaluate_model(
            transformer_model, "GraphTransformer_CIGIN", train_loader, valid_loader, test_loader, max_epochs
        ))
    
    # Print comparison results
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("BENCHMARK COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Model':<25} {'Parameters':<12} {'Test MSE':<10} {'Test MAE':<10} {'Test RMSE':<11} {'Time (min)':<10}")
        print(f"{'-'*80}")
        
        for result in results:
            print(f"{result['model_name']:<25} {result['parameters']:<12,} {result['test_mse']:<10.4f} "
                  f"{result['test_mae']:<10.4f} {result['test_rmse']:<11.4f} {result['training_time']/60:<10.2f}")
        
        # Calculate improvements
        if len(results) == 2:
            original_result = next(r for r in results if 'Original' in r['model_name'])
            transformer_result = next(r for r in results if 'GraphTransformer' in r['model_name'])
            
            mse_improvement = ((original_result['test_mse'] - transformer_result['test_mse']) / original_result['test_mse']) * 100
            mae_improvement = ((original_result['test_mae'] - transformer_result['test_mae']) / original_result['test_mae']) * 100
            rmse_improvement = ((original_result['test_rmse'] - transformer_result['test_rmse']) / original_result['test_rmse']) * 100
            
            print(f"\n{'='*80}")
            print("IMPROVEMENT ANALYSIS")
            print(f"{'='*80}")
            print(f"MSE Improvement: {mse_improvement:+.2f}% {'(Better)' if mse_improvement > 0 else '(Worse)'}")
            print(f"MAE Improvement: {mae_improvement:+.2f}% {'(Better)' if mae_improvement > 0 else '(Worse)'}")
            print(f"RMSE Improvement: {rmse_improvement:+.2f}% {'(Better)' if rmse_improvement > 0 else '(Worse)'}")
            
            param_ratio = transformer_result['parameters'] / original_result['parameters']
            print(f"Parameter Ratio: {param_ratio:.2f}x {'(More parameters)' if param_ratio > 1 else '(Fewer parameters)'}")
            
            time_ratio = transformer_result['training_time'] / original_result['training_time']
            print(f"Training Time Ratio: {time_ratio:.2f}x {'(Slower)' if time_ratio > 1 else '(Faster)'}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"runs/run-{project_name}/benchmark_results.csv", index=False)
        print(f"\nResults saved to: runs/run-{project_name}/benchmark_results.csv")
    
    else:
        print(f"\n{'='*50}")
        print("SINGLE MODEL RESULTS")
        print(f"{'='*50}")
        result = results[0]
        print(f"Model: {result['model_name']}")
        print(f"Parameters: {result['parameters']:,}")
        print(f"Test MSE: {result['test_mse']:.4f}")
        print(f"Test MAE: {result['test_mae']:.4f}")
        print(f"Test RMSE: {result['test_rmse']:.4f}")
        print(f"Training Time: {result['training_time']/60:.2f} minutes")

if __name__ == '__main__':
    main()
