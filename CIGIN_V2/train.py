from tqdm import tqdm
import torch
import numpy as np
import os

class EnhancedTrainer:
    def __init__(self, device=None, mean=0.0, std=1.0):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        self.mean = mean
        self.std = std
        
    def compute_metrics(self, model, data_loader):
        model.eval()
        total_loss = 0
        total_mae = 0
        with torch.no_grad():
            for solute, solvent, solute_len, solvent_len, labels in data_loader:
                solute = solute.to(self.device)
                solvent = solvent.to(self.device)
                solute_len = torch.tensor(solute_len, device=self.device)
                solvent_len = torch.tensor(solvent_len, device=self.device)
                targets = torch.tensor(labels, dtype=torch.float32, device=self.device)

                outputs, _ = model([solute, solvent, solute_len, solvent_len])
                
                # Denormalize
                outputs_denorm = outputs * self.std + self.mean
                targets_denorm = targets * self.std + self.mean
                
                total_loss += self.mse_loss(outputs_denorm, targets_denorm).item()
                total_mae += self.mae_loss(outputs_denorm, targets_denorm).item()
        return total_loss/len(data_loader), total_mae/len(data_loader)

    def train_epoch(self, model, optimizer, train_loader, grad_clip=None):
        model.train()
        epoch_loss = 0
        for solute, solvent, solute_len, solvent_len, labels in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            solute = solute.to(self.device)
            solvent = solvent.to(self.device)
            solute_len = torch.tensor(solute_len, device=self.device)
            solvent_len = torch.tensor(solvent_len, device=self.device)
            targets = torch.tensor(labels, dtype=torch.float32, device=self.device)

            outputs, interaction_map = model([solute, solvent, solute_len, solvent_len])
            
            loss = self.mse_loss(outputs, targets) + torch.norm(interaction_map, p=2) * 1e-4
            
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name, mean=0.0, std=1.0):
    trainer = EnhancedTrainer(mean=mean, std=std)
    best_val_loss = float('inf')
    save_dir = f"./runs/run-{project_name}/models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(max_epochs):
        train_loss = trainer.train_epoch(model, optimizer, train_loader)
        val_loss, val_mae = trainer.compute_metrics(model, valid_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{max_epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model.tar")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
            }, save_path)
            print(f"Saved new best model to {save_path}")

def get_metrics(model, data_loader, mean=0.0, std=1.0):
    trainer = EnhancedTrainer(mean=mean, std=std)
    return trainer.compute_metrics(model, data_loader)
