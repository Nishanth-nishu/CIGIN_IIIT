import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import os
import dgl
import warnings

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Import your existing modules
from model import CIGINModel
from train import train, get_metrics
from molecular_graph import get_graph_from_smile
from utils import get_len_matrix

class SolvationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get molecular graphs
        solute_graph = get_graph_from_smile(row['SoluteSMILES'])
        solvent_graph = get_graph_from_smile(row['SolventSMILES'])
        
        # Get molecule lengths (number of atoms)
        solute_len = solute_graph.number_of_nodes()
        solvent_len = solvent_graph.number_of_nodes()
        
        # Get target value
        target = float(row['delGsolv'])
        
        return solute_graph, solvent_graph, solute_len, solvent_len, target

def collate_fn(batch):
    """Custom collate function for batching molecular graphs using DGL's batch functionality"""
    solute_graphs, solvent_graphs, solute_lens, solvent_lens, targets = zip(*batch)
    
    # Batch the graphs using DGL's batch function
    batched_solute = dgl.batch(solute_graphs)
    batched_solvent = dgl.batch(solvent_graphs)
    
    # Create length matrices as described in CIGIN paper
    # The length matrix is used to mask interactions between different molecules in the batch
    solute_len_matrix = get_len_matrix(solute_lens)
    solvent_len_matrix = get_len_matrix(solvent_lens)
    
    return batched_solute, batched_solvent, solute_len_matrix, solvent_len_matrix, list(targets)

def main():
    # Load and preprocess data
    print("Loading dataset...")
    
    # Use the specific dataset URL you provided
    dataset_url = "https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/master/CIGIN_V2/data/whole_data.csv"
    
    try:
        print(f"Loading dataset from: {dataset_url}")
        df = pd.read_csv(dataset_url)
        print(f"Successfully loaded dataset from: {dataset_url}")
    except Exception as e:
        print(f"ERROR: Could not load dataset from {dataset_url}: {str(e)}")
        print("Please check the dataset URL or network connection.")
        return
    
    # Strip whitespace from column names first
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from all string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Columns after cleaning: {df.columns.tolist()}")
    
    # Verify required columns exist
    required_columns = ['SoluteName', 'SoluteSMILES', 'SolventName', 'SolventSMILES', 'delGsolv']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Remove any rows with missing SMILES or target values
    initial_len = len(df)
    df = df.dropna(subset=['SoluteSMILES', 'SolventSMILES', 'delGsolv'])
    print(f"Removed {initial_len - len(df)} rows with missing data")
    print(f"Final dataset size: {len(df)} samples")
    
    # Hyperparameters as mentioned in CIGIN paper
    node_input_dim = 42  # Based on atom features in paper
    edge_input_dim = 10  # Based on bond features in paper
    node_hidden_dim = 42
    edge_hidden_dim = 42
    num_step_message_passing = 6  # T=6 as mentioned in paper
    interaction = 'dot'  # dot product interaction as in paper
    batch_size = 32
    learning_rate = 0.001  # ADAM with default parameters
    max_epochs = 100
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 10-fold cross validation as described in CIGIN paper
    # Paper mentions: "10-fold cross validation scheme was used to assess the model due to the small size of the dataset"
    # "We made 5 such 10 cross validation splits and trained our model independently on each of them"
    
    all_fold_results = []
    
    # Run 5 independent 10-fold cross validation splits as in the paper
    for run in range(5):
        print(f"\n=== Cross Validation Run {run + 1}/5 ===")
        
        # Random split with different seed for each run
        kfold = KFold(n_splits=10, shuffle=True, random_state=42 + run)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
            print(f"\nRun {run + 1}, Fold {fold + 1}/10")
            
            # Split data - 9:1 ratio as mentioned in paper
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            
            print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
            
            # Create datasets
            train_dataset = SolvationDataset(train_df)
            val_dataset = SolvationDataset(val_df)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, collate_fn=collate_fn)
            
            # Initialize model - CIGIN with set2set as it performed best in paper
            model = CIGINModel(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_hidden_dim=edge_hidden_dim,
                num_step_message_passing=num_step_message_passing,
                interaction=interaction,
                num_step_set2_set=2,  # As mentioned in paper
                num_layer_set2set=1   # As mentioned in paper
            )
            
            # Move model to device
            model.to(device)
            
            # Initialize optimizer and scheduler as mentioned in paper
            # "ADAM optimizer with its default parameters as suggested by Kingma and Ba was used"
            # "The learning rate was decreased on plateau by a factor of 10^-1 from 10^-2 to 10^-5"
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.1, patience=10)
            
            # Create directory for this run and fold - MINIMAL FIX: Fixed directory naming
            run_fold_dir = f"./runs/run_{run + 1}_fold_{fold + 1}"
            os.makedirs(f"{run_fold_dir}/models", exist_ok=True)
            
            # Train model - MINIMAL FIX: Fixed project name format
            train(max_epochs, model, optimizer, scheduler, train_loader, 
                  val_loader, f"{run + 1}_fold_{fold + 1}")
            
            # Get final validation metrics
            model.eval()
            val_loss, val_mae = get_metrics(model, val_loader)
            fold_results.append({
                'run': run + 1,
                'fold': fold + 1, 
                'val_rmse': np.sqrt(val_loss),  # Convert MSE to RMSE
                'val_mae': val_mae
            })
            
            print(f"Run {run + 1}, Fold {fold + 1} - Val RMSE: {np.sqrt(val_loss):.4f}, Val MAE: {val_mae:.4f}")
        
        all_fold_results.extend(fold_results)
        
        # Calculate average for this run
        run_rmse = np.mean([r['val_rmse'] for r in fold_results])
        run_mae = np.mean([r['val_mae'] for r in fold_results])
        print(f"Run {run + 1} Average - RMSE: {run_rmse:.4f}, MAE: {run_mae:.4f}")
    
    # Calculate final statistics across all runs and folds
    all_rmse = [r['val_rmse'] for r in all_fold_results]
    all_mae = [r['val_mae'] for r in all_fold_results]
    
    final_rmse_mean = np.mean(all_rmse)
    final_rmse_std = np.std(all_rmse)
    final_mae_mean = np.mean(all_mae)
    final_mae_std = np.std(all_mae)
    
    print(f"\n=== Final Results (5 independent 10-fold CV runs) ===")
    print(f"Average RMSE: {final_rmse_mean:.4f} ± {final_rmse_std:.4f} kcal/mol")
    print(f"Average MAE: {final_mae_mean:.4f} ± {final_mae_std:.4f} kcal/mol")
    
    # Expected result from paper: RMSE of 0.57 ± 0.10 kcal/mol
    print(f"\nPaper reported RMSE: 0.57 ± 0.10 kcal/mol")
    print(f"Our result RMSE: {final_rmse_mean:.2f} ± {final_rmse_std:.2f} kcal/mol")
    
    # Save detailed results
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv("./cigin_5x10fold_cv_results.csv", index=False)
    
    # Save summary statistics
    summary = {
        'final_rmse_mean': final_rmse_mean,
        'final_rmse_std': final_rmse_std,
        'final_mae_mean': final_mae_mean,
        'final_mae_std': final_mae_std,
        'paper_rmse': 0.57,
        'paper_rmse_std': 0.10
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("./cigin_summary_results.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"- Detailed results: ./cigin_5x10fold_cv_results.csv")
    print(f"- Summary results: ./cigin_summary_results.csv")

if __name__ == "__main__":
    main()
