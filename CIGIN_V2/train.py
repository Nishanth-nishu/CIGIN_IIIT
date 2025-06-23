from tqdm import tqdm
import torch
import numpy as np
import os
import traceback

# Loss functions
loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

# CPU usage for now 
device = torch.device("cpu")

# Label normalization constants (compute these from the dataset!)
LABEL_MEAN = -3.82  
LABEL_STD = 1.21    

def normalize_labels(labels):
    return (labels - LABEL_MEAN) / LABEL_STD

def denormalize_labels(labels):
    return labels * LABEL_STD + LABEL_MEAN

def safe_tensor_conversion(data, target_device=device, dtype=torch.float32):
    try:
        if isinstance(data, torch.Tensor):
            return data.to(target_device).to(dtype)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data, dtype=dtype, device=target_device)
        else:
            return torch.tensor([data], dtype=dtype, device=target_device)
    except Exception as e:
        print(f"Error in tensor conversion: {e}")
        traceback.print_exc()
        return torch.zeros(1, dtype=dtype, device=target_device)

def validate_batch_data(batch_data):
    try:
        solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = batch_data
        if solute_graphs is None or solvent_graphs is None:
            return None
        if solute_graphs.number_of_nodes() == 0 or solvent_graphs.number_of_nodes() == 0:
            return None

        solute_lens = safe_tensor_conversion(solute_lens)
        solvent_lens = safe_tensor_conversion(solvent_lens)
        labels = safe_tensor_conversion(labels)

        if not torch.isfinite(labels).all():
            print("Skipping batch due to non-finite labels")
            return None

        labels = normalize_labels(labels)

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1:
            labels = labels.view(-1)

        if solute_lens.dim() == 0:
            solute_lens = solute_lens.unsqueeze(0)
        if solvent_lens.dim() == 0:
            solvent_lens = solvent_lens.unsqueeze(0)

        target_batch_size = labels.shape[0]
        if solute_lens.shape[0] != target_batch_size:
            solute_lens = solute_lens.repeat(target_batch_size)[:target_batch_size]
        if solvent_lens.shape[0] != target_batch_size:
            solvent_lens = solvent_lens.repeat(target_batch_size)[:target_batch_size]

        return solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels

    except Exception as e:
        print(f"Error validating batch data: {e}")
        traceback.print_exc()
        return None

def get_metrics(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Validation")):
            try:
                validated_data = validate_batch_data(batch_data)
                if validated_data is None:
                    print(f"Skipping validation batch {batch_idx+1}: invalid or empty data")
                    continue

                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data
                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)

                outputs, i_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])

                if outputs is None or outputs.numel() == 0 or labels.numel() == 0:
                    print(f" Skipping validation batch {batch_idx+1}: empty output or label")
                    continue

                outputs = outputs.view(-1)[:labels.shape[0]]
                labels = labels.view(-1)[:outputs.shape[0]]

                if outputs.numel() == 0 or labels.numel() == 0:
                    print(f"Skipping validation batch {batch_idx+1}: mismatch after slicing")
                    continue

                if not torch.isfinite(outputs).all() or not torch.isfinite(labels).all():
                    print(f"Skipping validation batch {batch_idx+1}: non-finite values detected")
                    continue

                loss = loss_fn(outputs, labels)
                mae_loss = mae_loss_fn(outputs, labels)

                if torch.isfinite(loss) and torch.isfinite(mae_loss):
                    valid_outputs.extend(denormalize_labels(outputs).cpu().numpy())
                    valid_labels.extend(denormalize_labels(labels).cpu().numpy())
                    valid_loss.append(loss.item())
                    valid_mae_loss.append(mae_loss.item())

            except Exception as e:
                print(f"Error in validation batch {batch_idx+1}: {e}")
                traceback.print_exc()
                continue

    if not valid_loss:
        print(" Warning: No valid validation batches processed")
        return float('inf'), float('inf')

    return np.mean(valid_loss), np.mean(valid_mae_loss)

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_val_loss = float('inf')
    os.makedirs(f"./runs/run-{project_name}/models", exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        successful_batches = 0

        print(f"\nEpoch {epoch+1}/{max_epochs}")

        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                optimizer.zero_grad()
                validated_data = validate_batch_data(batch_data)
                
                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data
                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)

                outputs, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])

                if outputs is None or outputs.numel() == 0:
                    continue

                outputs = outputs.view(-1)[:labels.shape[0]]
                labels = labels.view(-1)[:outputs.shape[0]]

                if outputs.numel() == 0 or labels.numel() == 0:
                    continue

                if not torch.isfinite(outputs).all() or not torch.isfinite(labels).all():
                    print(f" Non-finite values detected in batch {batch_idx+1}, skipping")
                    continue

                main_loss = loss_fn(outputs, labels)
                l1_norm = torch.norm(interaction_map, p=2) * 1e-4
                total_loss = main_loss + l1_norm

                if not torch.isfinite(total_loss):
                    print(f" Non-finite total loss at batch {batch_idx+1}, skipping")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += main_loss.item()
                successful_batches += 1

                print(f" Batch {batch_idx+1} successful - Loss: {main_loss.item():.4f}")
                print(f"Output sample: {denormalize_labels(outputs[:5]).detach().cpu().numpy()}")
                print(f"Label sample : {denormalize_labels(labels[:5]).detach().cpu().numpy()}")

            except Exception as e:
                print(f" Critical error in batch {batch_idx+1}: {e}")
                traceback.print_exc()
                continue

        avg_train_loss = epoch_loss / successful_batches if successful_batches else float('inf')
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        if successful_batches == 0:
            continue

        val_loss, mae_loss = get_metrics(model, valid_loader)
        print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, MAE: {mae_loss:.4f}")

        try:
            scheduler.step(val_loss)
        except Exception as scheduler_error:
            print(f" Scheduler error: {scheduler_error}")

        if val_loss < best_val_loss and val_loss != float('inf'):
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./runs/run-{project_name}/models/best_model.tar")
            print(f"💾 Best model saved with val loss: {val_loss:.4f}")

    print("Training complete.")
    return best_val_loss
