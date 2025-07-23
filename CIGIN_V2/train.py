from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

class SolvationTrainer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.loss_fn = torch.nn.MSELoss()
        self.mae_fn = torch.nn.L1Loss()
        
    def train_epoch(self, model, optimizer, train_loader):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc='Training')
        
        for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass
            outputs, interaction_map = model(
                [solute_graphs.to(self.device),
                 solvent_graphs.to(self.device),
                 torch.tensor(solute_lens).to(self.device),
                 torch.tensor(solvent_lens).to(self.device)]
            )
            
            # Loss computation
            targets = torch.tensor(labels).float().to(self.device)
            task_loss = self.loss_fn(outputs.squeeze(), targets)
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            loss = task_loss + l1_norm
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{epoch_loss/len(train_loader):.4f}'})
        
        return epoch_loss / len(train_loader)
    
    def validate(self, model, valid_loader):
        model.eval()
        val_loss, val_mae = 0, 0
        
        with torch.no_grad():
            for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in valid_loader:
                outputs, _ = model(
                    [solute_graphs.to(self.device),
                     solvent_graphs.to(self.device),
                     torch.tensor(solute_lens).to(self.device),
                     torch.tensor(solvent_lens).to(self.device)]
                )
                
                targets = torch.tensor(labels).float().to(self.device)
                val_loss += self.loss_fn(outputs.squeeze(), targets).item()
                val_mae += self.mae_fn(outputs.squeeze(), targets).item()
        
        return val_loss/len(valid_loader), val_mae/len(valid_loader)
    
    def train(self, model, optimizer, scheduler, train_loader, valid_loader, epochs=100):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, optimizer, train_loader)
            val_loss, val_mae = self.validate(model, valid_loader)
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}: '
                  f'Train Loss = {train_loss:.4f} | '
                  f'Val Loss = {val_loss:.4f} | '
                  f'Val MAE = {val_mae:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'best_model.pth')
