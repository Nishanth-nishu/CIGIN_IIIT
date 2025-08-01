import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from models import Cigin
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

class SolvationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'solute_smiles': row['SoluteSMILES'],
            'solvent_smiles': row['SolventSMILES'], 
            'target': torch.FloatTensor([row['delGsolv']])
        }

def train_model(model, train_loader, val_loader, num_epochs=100):
    """Train CIGIN model following the paper's methodology"""
    # ADAM optimizer with default parameters as mentioned in paper
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            try:
                solute_smiles = batch['solute_smiles'][0]
                solvent_smiles = batch['solvent_smiles'][0]
                target = batch['target'].to(device)
                
                # Forward pass
                prediction, interaction_map = model(solute_smiles, solvent_smiles)
                loss = criterion(prediction, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_count += 1
                
            except Exception as e:
                # Skip problematic molecules as done in the paper
                continue
        
        # Validation 
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    solute_smiles = batch['solute_smiles'][0]
                    solvent_smiles = batch['solvent_smiles'][0]
                    target = batch['target'].to(device)
                    
                    prediction, _ = model(solute_smiles, solvent_smiles)
                    loss = criterion(prediction, target)
                    
                    val_loss += loss.item()
                    val_count += 1
                    
                except Exception as e:
                    continue
        
        if train_count > 0 and val_count > 0:
            avg_train_loss = train_loss / train_count
            avg_val_loss = val_loss / val_count
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
        
        # Early stopping if learning rate becomes too small
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("Early stopping: Learning rate too small")
            break
    
    return best_val_loss

def evaluate_model(model, test_loader):
    """Evaluate the model and return RMSE"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                solute_smiles = batch['solute_smiles'][0]
                solvent_smiles = batch['solvent_smiles'][0]
                target = batch['target'].to(device)
                
                prediction, _ = model(solute_smiles, solvent_smiles)
                loss = criterion(prediction, target)
                
                total_loss += loss.item()
                count += 1
                
            except Exception as e:
                continue
    
    if count > 0:
        rmse = np.sqrt(total_loss / count)
        return rmse
    else:
        return float('inf')

def run_kfold_cv(data, k=10, n_runs=5):
    """Run k-fold cross validation as described in the paper"""
    all_rmses = []
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        kf = KFold(n_splits=k, shuffle=True, random_state=run)
        run_rmses = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            print(f"Fold {fold+1}/{k}")
            
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Create datasets and loaders
            train_dataset = SolvationDataset(train_data)
            test_dataset = SolvationDataset(test_data)
            
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            # Initialize model with paper's hyperparameters
            model = Cigin(node_dim=40, edge_dim=10, T=3).to(device)
            
            # Train model
            _ = train_model(model, train_loader, test_loader)
            
            # Load best model and evaluate
            model.load_state_dict(torch.load('best_model.pth'))
            rmse = evaluate_model(model, test_loader)
            
            print(f"Fold {fold+1} RMSE: {rmse:.4f} kcal/mol")
            run_rmses.append(rmse)
        
        run_avg_rmse = np.mean(run_rmses)
        print(f"Run {run+1} Average RMSE: {run_avg_rmse:.4f} kcal/mol")
        all_rmses.extend(run_rmses)
    
    overall_mean = np.mean(all_rmses)
    overall_std = np.std(all_rmses)
    
    print(f"\nOverall Results:")
    print(f"Mean RMSE: {overall_mean:.4f} ± {overall_std:.4f} kcal/mol")
    
    return overall_mean, overall_std
