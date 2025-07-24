import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Loss functions
loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

def evaluate_model(model, dataloader):
    """Evaluation function for compatibility with original code"""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in dataloader:
            # Convert inputs to tensors if they aren't already
            if not isinstance(solute_lens, torch.Tensor):
                solute_lens = torch.FloatTensor(solute_lens)
            if not isinstance(solvent_lens, torch.Tensor):
                solvent_lens = torch.FloatTensor(solvent_lens)
            if not isinstance(labels, torch.Tensor):
                labels = torch.FloatTensor(labels)
                
            outputs, _ , _= model([
                solute_graphs.to(device),
                solvent_graphs.to(device),
                solute_lens.to(device),
                solvent_lens.to(device)
            ])
            preds.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())
    return np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))

def get_metrics(model, data_loader):
    """Calculate MSE and MAE losses"""
    model.eval()
    valid_loss = []
    valid_mae_loss = []
    valid_outputs = []
    valid_labels = []
    
    with torch.no_grad():
        for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in data_loader:
            # Ensure all inputs are tensors
            if not isinstance(solute_lens, torch.Tensor):
                solute_lens = torch.FloatTensor(solute_lens)
            if not isinstance(solvent_lens, torch.Tensor):
                solvent_lens = torch.FloatTensor(solvent_lens)
            if not isinstance(labels, torch.Tensor):
                labels = torch.FloatTensor(labels)
                
            # Move data to device
            inputs = [
                solute_graphs.to(device),
                solvent_graphs.to(device),
                solute_lens.to(device),
                solvent_lens.to(device)
            ]
            labels = labels.to(device)
            
            # Forward pass
            outputs, _, _ = model(inputs)

            
            # Calculate losses
            loss = loss_fn(outputs, labels)
            mae_loss = mae_loss_fn(outputs, labels)
            
            # Store results
            valid_loss.append(loss.cpu().item())
            valid_mae_loss.append(mae_loss.cpu().item())
            valid_outputs.extend(outputs.cpu().numpy().flatten())
            valid_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate mean losses
    loss = np.mean(valid_loss)
    mae_loss = np.mean(valid_mae_loss)
    return loss, mae_loss

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    """Main training loop"""
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = []
        
        # Initialize progress bar
        tq_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for samples in tq_loader:
            optimizer.zero_grad()
            
            # Convert and move data to device
            solute_graphs = samples[0].to(device)
            solvent_graphs = samples[1].to(device)
            
            # Handle length matrices
            solute_lens = samples[2]
            solvent_lens = samples[3]
            if not isinstance(solute_lens, torch.Tensor):
                solute_lens = torch.FloatTensor(solute_lens)
            if not isinstance(solvent_lens, torch.Tensor):
                solvent_lens = torch.FloatTensor(solvent_lens)
            solute_lens = solute_lens.to(device)
            solvent_lens = solvent_lens.to(device)
            
            # Handle labels
            labels = samples[4]
            if not isinstance(labels, torch.Tensor):
                labels = torch.FloatTensor(labels)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])

            
            # Calculate loss with L1 regularization
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            loss = loss_fn(outputs, labels) + l1_norm
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update running loss (without regularization term)
            running_loss.append((loss - l1_norm).cpu().item())
            
            # Update progress bar
            tq_loader.set_postfix(loss=np.mean(running_loss))
        
        # Validation phase
        val_loss, mae_loss = get_metrics(model, valid_loader)
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {np.mean(running_loss):.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {mae_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./runs/run-{project_name}/models/best_model.pt")
            print(f"Saved new best model with Val Loss: {best_val_loss:.4f}")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    print("This module contains training utilities and should be imported")
