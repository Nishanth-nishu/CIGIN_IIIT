from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def evaluate_model(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for samples in dataloader:
            outputs, _ = model(
                [samples[0].to(device), samples[1].to(device),
                 samples[2].to(device), samples[3].to(device)],
                return_attention=False
            )
            preds.extend(outputs.cpu().numpy())
            targets.extend(samples[4])
    return np.sqrt(np.mean((np.array(preds) - np.array(targets))**2)

def get_metrics(model, data_loader):
    model.eval()
    valid_outputs, valid_labels = [], []
    valid_loss, valid_mae_loss = [], []
    
    with torch.no_grad():
        for samples in data_loader:
            outputs, _ = model(
                [samples[0].to(device), samples[1].to(device),
                 torch.tensor(samples[2]).to(device),
                 torch.tensor(samples[3]).to(device)],
                return_attention=False
            )
            loss = loss_fn(outputs, torch.tensor(samples[4]).to(device).float())
            mae_loss = mae_loss_fn(outputs, torch.tensor(samples[4]).float())
            
            valid_outputs.extend(outputs.cpu().numpy())
            valid_labels.extend(samples[4])
            valid_loss.append(loss.item())
            valid_mae_loss.append(mae_loss.item())
    
    return np.mean(valid_loss), np.mean(valid_mae_loss)

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for samples in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass with attention
            outputs, interaction_map, attn_weights = model(
                [samples[0].to(device), samples[1].to(device),
                 torch.tensor(samples[2]).to(device),
                 torch.tensor(samples[3]).to(device)],
                return_attention=True
            )
            
            # Loss calculation
            task_loss = loss_fn(outputs, torch.tensor(samples[4]).to(device).float())
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            attn_reg = 0.01 * torch.mean(torch.sum(attn_weights**2, dim=-1))  # Attention sparsity
            
            total_loss = task_loss + l1_norm + attn_reg
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({'train_loss': f'{epoch_loss/len(train_loader):.4f}'})
        
        # Validation
        val_loss, val_mae = get_metrics(model, valid_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss = {epoch_loss/len(train_loader):.4f} | '
              f'Val Loss = {val_loss:.4f} | '
              f'Val MAE = {val_mae:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'./runs/run-{project_name}/models/best_model.tar')