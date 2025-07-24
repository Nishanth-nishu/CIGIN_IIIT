import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dgl import DGLGraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda.amp as amp

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Add this function to your existing train.py (keep everything else the same)
def evaluate_model(model, dataloader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in dataloader:
            outputs, _ = model([
                solute_graphs.to(device),
                solvent_graphs.to(device),
                solute_lens.to(device),
                solvent_lens.to(device)
            ])
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels)
    return np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))

class MultiTaskLossWrapper(nn.Module):
    """Adaptive loss weighting for multi-task learning"""
    def __init__(self, task_num=3):
        super().__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, preds, targets):
        # Main task loss (ΔG prediction)
        mse_loss = self.mse(preds[0], targets[0])
        mae_loss = self.mae(preds[0], targets[0])
        
        # Auxiliary task losses (logP, TPSA, QED)
        aux_loss = 0
        if len(targets) > 1 and targets[1] is not None:
            for i in range(self.task_num):
                aux_loss += torch.exp(-self.log_vars[i]) * self.mse(preds[1][:,i], targets[1][:,i]) + self.log_vars[i]
        
        return {
            'total': mse_loss + 0.3 * aux_loss,  # Weighted sum
            'mse': mse_loss,
            'mae': mae_loss,
            'aux': aux_loss
        }

def get_metrics(model, data_loader, return_preds=False):
    """Enhanced evaluation with optional prediction returns"""
    model.eval()
    total_loss = {'mse': 0, 'mae': 0, 'aux': 0}
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in data_loader:
            inputs = [
                solute_graphs.to(device),
                solvent_graphs.to(device),
                solute_lens.to(device),
                solvent_lens.to(device)
            ]
            labels = labels.to(device)
            
            # Forward pass
            with amp.autocast(enabled=use_cuda):
                main_pred, aux_pred, _ = model(inputs)
                loss_fn = MultiTaskLossWrapper()
                losses = loss_fn((main_pred, aux_pred), (labels, None))
            
            # Accumulate metrics
            for k in total_loss:
                total_loss[k] += losses[k].item() * len(labels)
            
            if return_preds:
                all_preds.extend(main_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate averages
    num_samples = len(data_loader.dataset)
    metrics = {k: v / num_samples for k, v in total_loss.items()}
    
    if return_preds:
        return metrics, (np.array(all_preds), np.array(all_labels))
    return metrics

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    """Enhanced training loop with multiple improvements"""
    best_val_loss = float('inf')
    loss_fn = MultiTaskLossWrapper()
    scaler = amp.GradScaler(enabled=use_cuda)
    
    # Training statistics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = {'mse': 0, 'mae': 0, 'aux': 0}
        total_samples = 0
        
        # Gradient accumulation
        accum_steps = 4
        optimizer.zero_grad()
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, samples in enumerate(tepoch):
                inputs = [
                    samples[0].to(device),
                    samples[1].to(device),
                    samples[2].to(device),
                    samples[3].to(device)
                ]
                labels = samples[4].to(device)
                batch_size = labels.shape[0]
                total_samples += batch_size
                
                # Mixed precision forward
                with amp.autocast(enabled=use_cuda):
                    main_pred, aux_pred, i_map = model(inputs)
                    l1_norm = torch.norm(i_map, p=2) * 1e-4
                    losses = loss_fn((main_pred, aux_pred), (labels, None))
                    loss = losses['total'] / accum_steps + l1_norm
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation update
                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Update running losses
                for k in running_loss:
                    running_loss[k] += losses[k].item() * batch_size
                
                # Progress bar update
                tepoch.set_postfix({
                    'loss': f"{running_loss['mse']/total_samples:.4f}",
                    'mae': f"{running_loss['mae']/total_samples:.4f}"
                })
        
        # Calculate epoch metrics
        train_metrics = {k: v/total_samples for k, v in running_loss.items()}
        val_metrics = get_metrics(model, valid_loader)
        
        # Update scheduler
        scheduler.step(val_metrics['mse'])
        
        # Store history
        history['train_loss'].append(train_metrics['mse'])
        history['val_loss'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{max_epochs}:")
        print(f"Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
        print(f"Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f}")
        print(f"LR: {history['lr'][-1]:.2e}")
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'metrics': val_metrics,
                'history': history
            }, f"./runs/run-{project_name}/models/best_model.tar")
            print(f"New best model saved with Val MSE: {best_val_loss:.4f}")
    
    return history

def load_best_model(model, project_name):
    """Load the best saved model"""
    checkpoint = torch.load(f"./runs/run-{project_name}/models/best_model.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['history']

if __name__ == '__main__':
    # Example usage (would normally be called from main.py)
    print("This module contains training utilities and should be imported")
