from tqdm import tqdm
import torch
import numpy as np
import os

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

# Force CPU usage
device = torch.device("cpu")

def safe_tensor_conversion(data, target_device=device, dtype=torch.float32):
    """Safely convert data to tensor with proper error handling"""
    try:
        if isinstance(data, torch.Tensor):
            return data.to(target_device).to(dtype)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data, dtype=dtype, device=target_device)
        else:
            return torch.tensor([data], dtype=dtype, device=target_device)
    except Exception as e:
        print(f"Error in tensor conversion: {e}")
        return torch.zeros(1, dtype=dtype, device=target_device)

def validate_batch_data(batch_data):
    """Validate and fix batch data before processing"""
    try:
        solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = batch_data
        
        # Ensure graphs are valid
        if solute_graphs is None or solvent_graphs is None:
            return None
            
        # Check if graphs have nodes
        if solute_graphs.number_of_nodes() == 0 or solvent_graphs.number_of_nodes() == 0:
            return None
            
        # Safely convert tensors
        solute_lens = safe_tensor_conversion(solute_lens)
        solvent_lens = safe_tensor_conversion(solvent_lens)
        labels = safe_tensor_conversion(labels)
        
        # Ensure batch consistency
        batch_size = max(1, len(labels) if len(labels.shape) > 0 else 1)
        
        # Reshape if needed
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if labels.dim() > 1:
            labels = labels.view(-1)
            
        if solute_lens.dim() == 0:
            solute_lens = solute_lens.unsqueeze(0)
        if solvent_lens.dim() == 0:
            solvent_lens = solvent_lens.unsqueeze(0)
            
        # Ensure all have same batch size
        target_batch_size = labels.shape[0]
        if solute_lens.shape[0] != target_batch_size:
            solute_lens = solute_lens.repeat(target_batch_size)[:target_batch_size]
        if solvent_lens.shape[0] != target_batch_size:
            solvent_lens = solvent_lens.repeat(target_batch_size)[:target_batch_size]
            
        return solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels
        
    except Exception as e:
        print(f"Error validating batch data: {e}")
        return None

def get_metrics(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Validation"):
            try:
                # Validate batch data
                validated_data = validate_batch_data(batch_data)
                if validated_data is None:
                    continue
                    
                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data
                
                # Ensure everything is on the correct device
                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)
                
                # Forward pass with error handling
                try:
                    outputs, i_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
                except Exception as model_error:
                    print(f"Model forward pass error: {model_error}")
                    continue
                
                # Ensure outputs and labels have compatible shapes
                if outputs.numel() == 0 or labels.numel() == 0:
                    continue
                    
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                
                # Take minimum size to avoid dimension mismatch
                min_size = min(outputs.shape[0], labels.shape[0])
                if min_size == 0:
                    continue
                    
                outputs = outputs[:min_size]
                labels = labels[:min_size]
                
                loss = loss_fn(outputs, labels)
                mae_loss = mae_loss_fn(outputs, labels)
                
                # Check for valid loss values
                if torch.isfinite(loss) and torch.isfinite(mae_loss):
                    valid_outputs.extend(outputs.cpu().detach().numpy().tolist())
                    valid_loss.append(loss.cpu().detach().numpy())
                    valid_mae_loss.append(mae_loss.cpu().detach().numpy())
                    valid_labels.extend(labels.cpu().detach().numpy().tolist())
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    if len(valid_loss) == 0:
        print("Warning: No valid validation batches processed")
        return float('inf'), float('inf')
    
    loss = np.mean(np.array(valid_loss))
    mae_loss = np.mean(np.array(valid_mae_loss))
    return loss, mae_loss

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_val_loss = float('inf')
    
    # Create directories if they don't exist
    os.makedirs(f"./runs/run-{project_name}/models", exist_ok=True)
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = []
        epoch_loss = 0.0
        num_batches = 0
        successful_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                # Clear gradients
                optimizer.zero_grad()
                
                # Validate batch data
                validated_data = validate_batch_data(batch_data)
                if validated_data is None:
                    print(f"Skipping invalid batch {batch_idx}")
                    continue
                    
                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data
                
                # Ensure everything is on the correct device
                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)
                
                # Debug info for first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"\nDebug info for first batch:")
                    print(f"Solute graph nodes: {solute_graphs.number_of_nodes()}")
                    print(f"Solvent graph nodes: {solvent_graphs.number_of_nodes()}")
                    print(f"Solute lens shape: {solute_lens.shape}")
                    print(f"Solvent lens shape: {solvent_lens.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Labels values: {labels}")
                
                # Forward pass with comprehensive error handling
                try:
                    outputs, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
                except Exception as model_error:
                    print(f"Model forward pass error in batch {batch_idx}: {model_error}")
                    continue
                
                # Validate outputs
                if outputs is None or interaction_map is None:
                    print(f"Model returned None outputs for batch {batch_idx}")
                    continue
                    
                if outputs.numel() == 0 or labels.numel() == 0:
                    print(f"Empty outputs or labels in batch {batch_idx}")
                    continue
                
                # Ensure compatible shapes
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                
                # Take minimum size to avoid dimension mismatch
                min_size = min(outputs.shape[0], labels.shape[0])
                if min_size == 0:
                    print(f"Zero size after shape matching in batch {batch_idx}")
                    continue
                    
                outputs = outputs[:min_size]
                labels = labels[:min_size]
                
                # Calculate losses with safety checks
                try:
                    main_loss = loss_fn(outputs, labels)
                    
                    # L2 regularization on interaction map with safety check
                    if interaction_map.numel() > 0:
                        l2_reg = torch.norm(interaction_map, p=2) * 1e-4
                    else:
                        l2_reg = torch.tensor(0.0, device=device)
                    
                    total_loss = main_loss + l2_reg
                    
                    # Check for valid loss
                    if not torch.isfinite(total_loss):
                        print(f"Non-finite loss in batch {batch_idx}: {total_loss}")
                        continue
                    
                except Exception as loss_error:
                    print(f"Loss calculation error in batch {batch_idx}: {loss_error}")
                    continue
                
                # Backward pass with error handling
                try:
                    total_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                except Exception as backward_error:
                    print(f"Backward pass error in batch {batch_idx}: {backward_error}")
                    continue
                
                # Track successful batch
                batch_loss = main_loss.item()
                running_loss.append(batch_loss)
                epoch_loss += batch_loss
                successful_batches += 1
                
                if batch_idx == 0 and epoch == 0:
                    print(f"First batch successful - Loss: {batch_loss:.4f}")
                    print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                    print(f"Output values: {outputs.detach().cpu().numpy()}")
                
            except Exception as e:
                print(f"\nCritical error in training batch {batch_idx}:")
                print(f"Error: {e}")
                continue
            
            finally:
                # Ensure we increment num_batches for attempted batches
                num_batches += 1
        
        # Calculate average training loss
        if successful_batches > 0:
            avg_train_loss = epoch_loss / successful_batches
        else:
            print(f"Warning: No successful training batches in epoch {epoch + 1}")
            avg_train_loss = float('inf')
        
        # Validation with error handling
        try:
            print("Running validation...")
            val_loss, mae_loss = get_metrics(model, valid_loader)
        except Exception as val_error:
            print(f"Validation error: {val_error}")
            val_loss, mae_loss = float('inf'), float('inf')
        
        # Update learning rate scheduler
        try:
            scheduler.step(val_loss)
        except Exception as scheduler_error:
            print(f"Scheduler error: {scheduler_error}")
        
        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae_loss:.4f}")
        print(f"Successful batches: {successful_batches}/{num_batches}")
        
        # Save best model
        if val_loss < best_val_loss and val_loss != float('inf'):
            best_val_loss = val_loss
            try:
                model_path = f"./runs/run-{project_name}/models/best_model.tar"
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
        
        # Early stopping checks
        if successful_batches == 0:
            print("Error: No successful training batches, stopping training")
            break
            
        if val_loss == float('inf') and epoch > 0:
            print("Warning: Validation loss is infinite")
        
        # Memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss