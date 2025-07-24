import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import logging
import os

# Loss functions
mae_criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()

# ---------------- Logger Setup ----------------
def initialize_logger(log_file="training.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode='w'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def log_metrics_to_file(epoch, train_loss, train_mae, val_loss, val_mae):
    logging.info(f"Epoch {epoch}:")
    logging.info(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
    logging.info(f"  Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

def log_metrics_to_wandb(epoch, train_loss, train_mae, val_loss, val_mae):
    wandb.log({
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Train MAE": train_mae,
        "Val Loss": val_loss,
        "Val MAE": val_mae
    })

# ---------------- Training Function ----------------
def train(max_epochs, model, optimizer, scheduler, train_loader, val_loader, project_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    wandb.init(project=project_name)
    wandb.watch(model)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for batch in progress:
            try:
                batch = [item.to(device) if hasattr(item, 'to') else item for item in batch]
                outputs, interaction_map = model(batch)

                labels = batch[-1].to(device)

                loss = mse_criterion(outputs, labels)
                mae = mae_criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mae += mae.item()
            except Exception as e:
                logging.error(f"Error in training batch: {e}")
                continue

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_mae = epoch_mae / len(train_loader)

        # Validation
        val_loss, val_mae = evaluate_model(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        log_metrics_to_file(epoch, avg_train_loss, avg_train_mae, val_loss, val_mae)
        log_metrics_to_wandb(epoch, avg_train_loss, avg_train_mae, val_loss, val_mae)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"{project_name}_best_model.pth")

        logging.info(f"Finished Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")

    logging.info(f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

# ---------------- Evaluation Functions ----------------
def evaluate_model(model, data_loader, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [item.to(device) if hasattr(item, 'to') else item for item in batch]
                outputs, _ = model(batch)
                labels = batch[-1].to(device)

                loss = mse_criterion(outputs, labels)
                mae = mae_criterion(outputs, labels)

                total_loss += loss.item()
                total_mae += mae.item()

                if return_predictions:
                    predictions.append(outputs.cpu())
                    ground_truth.append(labels.cpu())
            except Exception as e:
                logging.error(f"Error in validation batch: {e}")
                continue

    avg_loss = total_loss / len(data_loader)
    avg_mae = total_mae / len(data_loader)

    if return_predictions:
        return avg_loss, avg_mae, torch.cat(predictions), torch.cat(ground_truth)
    return avg_loss, avg_mae

def get_metrics(model, data_loader, device, return_predictions=False):
    return evaluate_model(model, data_loader, device, return_predictions)
