from tqdm import tqdm
import torch
import numpy as np

class EnhancedTrainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        
    def compute_metrics(self, model, data_loader):
        model.eval()
        total_loss = 0
        total_mae = 0
        with torch.no_grad():
            for solute, solvent, solute_len, solvent_len, labels in data_loader:
                # Convert numpy arrays to tensors first
                solute_len = torch.tensor(solute_len).to(self.device)
                solvent_len = torch.tensor(solvent_len).to(self.device)
                
                outputs, _ = model([
                    solute.to(self.device),
                    solvent.to(self.device),
                    solute_len,
                    solvent_len
                ])
                targets = torch.tensor(labels).float().to(self.device)
                total_loss += self.mse_loss(outputs, targets).item()
                total_mae += self.mae_loss(outputs, targets).item()
        return total_loss/len(data_loader), total_mae/len(data_loader)

    def train_epoch(self, model, optimizer, train_loader, grad_clip=None):
        model.train()
        epoch_loss = 0
        for solute, solvent, solute_len, solvent_len, labels in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # Convert numpy arrays to tensors first
            solute_len = torch.tensor(solute_len).to(self.device)
            solvent_len = torch.tensor(solvent_len).to(self.device)
            
            # Forward pass
            outputs, interaction_map = model([
                solute.to(self.device),
                solvent.to(self.device),
                solute_len,
                solvent_len
            ])
            
            # Loss computation
            targets = torch.tensor(labels).float().to(self.device)
            loss = self.mse_loss(outputs, targets) + torch.norm(interaction_map, p=2) * 1e-4
            
            # Backward pass
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    trainer = EnhancedTrainer()
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # Training
        train_loss = trainer.train_epoch(model, optimizer, train_loader)
        
        # Validation
        val_loss, val_mae = trainer.compute_metrics(model, valid_loader)
        scheduler.step(val_loss)
        
        # Logging
        print(f"Epoch {epoch+1}/{max_epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}")
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
            }, f"./runs/run-{project_name}/models/best_model.tar")

def get_metrics(model, data_loader):
    trainer = EnhancedTrainer()
    return trainer.compute_metrics(model, data_loader)
