from tqdm import tqdm
import torch
import numpy as np
import os
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SolvationTrainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        self.best_val_loss = float('inf')
        
    def prepare_batch(self, batch):
        """Convert batch data to proper tensors and device"""
        solute, solvent, solute_len, solvent_len, labels = batch
        return (
            solute.to(self.device),
            solvent.to(self.device),
            torch.tensor(solute_len, device=self.device),
            torch.tensor(solvent_len, device=self.device),
            torch.tensor(labels, dtype=torch.float32, device=self.device)
        )

    def compute_loss(self, outputs, targets, interaction_map):
        """Enhanced loss calculation with multiple regularization terms"""
        # Main MSE loss
        task_loss = self.mse_loss(outputs, targets)
        
        # Original L1 regularization on interaction map
        l1_norm = torch.norm(interaction_map, p=2) * 1e-4
        
        # Additional regularization terms
        bond_attention_loss = torch.mean(interaction_map.pow(2)) * 1e-3  # Penalize large attention values
        output_std_loss = torch.std(outputs) * 1e-2  # Encourage stable predictions
        
        return task_loss + l1_norm + bond_attention_loss + output_std_loss

    def train_epoch(self, model, optimizer, train_loader, grad_clip=1.0):
        """Enhanced training loop with gradient clipping and loss tracking"""
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Prepare batch data
            solute, solvent, solute_len, solvent_len, targets = self.prepare_batch(batch)
            
            # Forward pass
            outputs, interaction_map = model([solute, solvent, solute_len, solvent_len])
            
            # Calculate loss
            loss = self.compute_loss(outputs, targets, interaction_map)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # Update tracking
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_loss/len(progress_bar):.4f}'})
            
        return total_loss / len(train_loader)

    def evaluate(self, model, data_loader):
        """Comprehensive model evaluation"""
        model.eval()
        total_loss = 0
        total_mae = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                solute, solvent, solute_len, solvent_len, targets = self.prepare_batch(batch)
                
                outputs, interaction_map = model([solute, solvent, solute_len, solvent_len])
                
                # Calculate metrics
                total_loss += self.mse_loss(outputs, targets).item() * len(targets)
                total_mae += self.mae_loss(outputs, targets).item() * len(targets)
                total_samples += len(targets)
                
        return total_loss/total_samples, total_mae/total_samples

    def save_checkpoint(self, model, optimizer, epoch, val_loss, save_dir):
        """Enhanced checkpoint saving"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_model.tar")
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
                'best_loss': self.best_val_loss,
                'config': model.config if hasattr(model, 'config') else {}
            }, save_path)
            return save_path
        return None

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    """Main training function with enhanced features"""
    trainer = SolvationTrainer()
    
    # Training metrics tracking
    train_history = {'loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(max_epochs):
        # Training phase
        train_loss = trainer.train_epoch(model, optimizer, train_loader)
        train_history['loss'].append(train_loss)
        
        # Validation phase
        val_loss, val_mae = trainer.evaluate(model, valid_loader)
        train_history['val_loss'].append(val_loss)
        train_history['val_mae'].append(val_mae)
        
        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{max_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        save_path = trainer.save_checkpoint(
            model, optimizer, epoch, val_loss, 
            f"./runs/run-{project_name}/models"
        )
        if save_path:
            print(f"  Saved best model to {save_path} (loss: {val_loss:.4f})")
    
    return train_history

def get_metrics(model, data_loader):
    """Get evaluation metrics for a model"""
    trainer = SolvationTrainer()
    return trainer.evaluate(model, data_loader)
