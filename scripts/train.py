from tqdm import tqdm
import torch
import numpy as np

loss_fn = torch.nn.MSELoss()
mae_fn = torch.nn.L1Loss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_metrics(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    losses, mae_losses = [], []

    with torch.no_grad():
        for solute, solvent, labels in tqdm(data_loader):
            outputs, _ = model(solute, solvent)
            labels = torch.tensor(labels).to(device).float()

            loss = loss_fn(outputs, labels)
            mae = mae_fn(outputs, labels)

            losses.append(loss.item())
            mae_losses.append(mae.item())

            all_preds += outputs.cpu().numpy().tolist()
            all_labels += labels.cpu().numpy().tolist()

    return np.mean(losses), np.mean(mae_losses)


def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, save_path):
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        model.train()
        running_losses = []

        for solute, solvent, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            outputs, interaction_map = model(solute, solvent)
            labels = torch.tensor(labels).to(device).float()

            l2_penalty = torch.norm(interaction_map, p=2) * 1e-4
            loss = loss_fn(outputs, labels) + l2_penalty

            loss.backward()
            optimizer.step()

            running_losses.append((loss - l2_penalty).item())

        val_loss, val_mae = get_metrics(model, valid_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {np.mean(running_losses):.4f} | Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("✅ Model saved.")
