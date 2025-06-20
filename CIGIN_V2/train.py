from tqdm import tqdm
import torch
import numpy as np
import os
import traceback

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

# Force CPU usage
device = torch.device("cpu")

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

        batch_size = max(1, len(labels) if len(labels.shape) > 0 else 1)

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if labels.dim() > 1:
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
        for batch_data in tqdm(data_loader, desc="Validation"):
            try:
                validated_data = validate_batch_data(batch_data)
                if validated_data is None:
                    continue

                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data

                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)

                try:
                    outputs, i_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
                except Exception as model_error:
                    print(f"Model forward pass error: {model_error}")
                    traceback.print_exc()
                    continue

                if outputs.numel() == 0 or labels.numel() == 0:
                    continue

                outputs = outputs.view(-1)
                labels = labels.view(-1)

                min_size = min(outputs.shape[0], labels.shape[0])
                if min_size == 0:
                    continue

                outputs = outputs[:min_size]
                labels = labels[:min_size]

                loss = loss_fn(outputs, labels)
                mae_loss = mae_loss_fn(outputs, labels)

                if torch.isfinite(loss) and torch.isfinite(mae_loss):
                    valid_outputs.extend(outputs.cpu().detach().numpy().tolist())
                    valid_loss.append(loss.cpu().detach().numpy())
                    valid_mae_loss.append(mae_loss.cpu().detach().numpy())
                    valid_labels.extend(labels.cpu().detach().numpy().tolist())

            except Exception as e:
                print(f"Error in validation batch: {e}")
                traceback.print_exc()
                continue

    if len(valid_loss) == 0:
        print("Warning: No valid validation batches processed")
        return float('inf'), float('inf')

    loss = np.mean(np.array(valid_loss))
    mae_loss = np.mean(np.array(valid_mae_loss))
    return loss, mae_loss

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_val_loss = float('inf')

    os.makedirs(f"./runs/run-{project_name}/models", exist_ok=True)

    for epoch in range(max_epochs):

        model.train()
        running_loss = []
        epoch_loss = 0.0
        num_batches = 0
        successful_batches = 0

        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print(f"Starting Epoch {epoch+1} with {len(train_loader)} batches")

        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                print(f"Processing batch {batch_idx+1}")

                optimizer.zero_grad()

                validated_data = validate_batch_data(batch_data)
                if validated_data is None:
                    print(f"Skipping invalid batch {batch_idx}")
                    continue

                solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = validated_data

                solute_graphs = solute_graphs.to(device)
                solvent_graphs = solvent_graphs.to(device)
                solute_lens = solute_lens.to(device)
                solvent_lens = solvent_lens.to(device)
                labels = labels.to(device)

                if batch_idx == 0 and epoch == 0:
                    print(f"\nDebug info for first batch:")
                    print(f"Solute graph nodes: {solute_graphs.number_of_nodes()}")
                    print(f"Solvent graph nodes: {solvent_graphs.number_of_nodes()}")
                    print(f"Solute lens shape: {solute_lens.shape}")
                    print(f"Solvent lens shape: {solvent_lens.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Labels values: {labels}")

                try:
                    print("🔁 Calling model forward pass...")
                    outputs, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
                    print("✅ Model forward pass completed.")

                except Exception as model_error:
                    print(f"Model forward pass error in batch {batch_idx}: {model_error}")
                    traceback.print_exc()
                    continue

                if outputs is None or interaction_map is None:
                    print(f"Model returned None outputs for batch {batch_idx}")
                    continue

                if outputs.numel() == 0 or labels.numel() == 0:
                    print(f"Empty outputs or labels in batch {batch_idx}")
                    continue

                outputs = outputs.view(-1)
                labels = labels.view(-1)
                if torch.isnan(outputs).any() or torch.isnan(labels).any():
                    print(f"❌ NaN detected in outputs or labels - skipping batch {batch_idx+1}")
                    continue


                min_size = min(outputs.shape[0], labels.shape[0])
                if min_size == 0:
                    print(f"Zero size after shape matching in batch {batch_idx}")
                    continue

                outputs = outputs[:min_size]
                labels = labels[:min_size]

                try:
                    main_loss = loss_fn(outputs, labels)

                    if interaction_map.numel() > 0:
                        l2_reg = torch.norm(interaction_map, p=2) * 1e-4
                    else:
                        l2_reg = torch.tensor(0.0, device=device)

                    total_loss = main_loss + l2_reg

                    if not torch.isfinite(total_loss):
                        print(f"Non-finite loss in batch {batch_idx}: {total_loss}")
                        continue

                except Exception as loss_error:
                    print(f"Loss calculation error in batch {batch_idx}: {loss_error}")
                    traceback.print_exc()
                    continue

                try:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except Exception as backward_error:
                    print(f"Backward pass error in batch {batch_idx}: {backward_error}")
                    traceback.print_exc()
                    continue

                batch_loss = main_loss.item()
                running_loss.append(batch_loss)
                epoch_loss += batch_loss
                successful_batches += 1
                print(f"✅ Batch {batch_idx+1} successful - Loss: {batch_loss:.4f}")
                print(f"   Outputs: {outputs.detach().cpu().numpy()}")
                print(f"   Labels : {labels.detach().cpu().numpy()}")


                if batch_idx == 0 and epoch == 0:
                    print(f"First batch successful - Loss: {batch_loss:.4f}")
                    print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                    print(f"Output values: {outputs.detach().cpu().numpy()}")
                print(f"🔁 Finished {num_batches} total batch iterations in this epoch")


            except Exception as e:
                print(f"\nCritical error in training batch {batch_idx}:")
                print(f"Error: {e}")
                traceback.print_exc()
                continue

            finally:
                num_batches += 1

        avg_train_loss = epoch_loss / successful_batches if successful_batches > 0 else float('inf')
        print(f"🔁 Finished {successful_batches} successful batches out of {num_batches} in this epoch")
        if successful_batches == 0:
            print("Warning: No successful batches in this epoch, skipping validation")
            continue
        print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")
        if avg_train_loss == float('inf'):
            print("Warning: Average training loss is infinite, skipping validation")
            continue
        print("🔁 Starting validation...")

        try:
            print("Running validation...")
            val_loss, mae_loss = get_metrics(model, valid_loader)
        except Exception as val_error:
            print(f"Validation error: {val_error}")
            traceback.print_exc()
            val_loss, mae_loss = float('inf'), float('inf')

        try:
            scheduler.step(val_loss)
        except Exception as scheduler_error:
            print(f"Scheduler error: {scheduler_error}")
            traceback.print_exc()

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae_loss:.4f}")
        print(f"Successful batches: {successful_batches}/{num_batches}")

        if val_loss < best_val_loss and val_loss != float('inf'):
            best_val_loss = val_loss
            try:
                model_path = f"./runs/run-{project_name}/models/best_model.tar"
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
                traceback.print_exc()

        if successful_batches == 0:
            print("Error: No successful training batches, stopping training")
            continue

        if val_loss == float('inf') and epoch > 0:
            print("Warning: Validation loss is infinite")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Finished all {successful_batches} batches in Epoch {epoch+1}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    if best_val_loss == float('inf'):
        print("Warning: Best validation loss is infinite, training may have failed.")
        return None
    print("✅ Training loop completed fully.")

    return best_val_loss
