import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

from utils import save_model, MAE, initialize_logger


def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    logger = initialize_logger(project_name)
    wandb.init(project=project_name)
    wandb.watch(model)

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        print(f"Epoch {epoch}/{max_epochs}:")
        for batch in tqdm(train_loader):
            solute_graphs, solvent_graphs, labels = batch
            solute_lens = solute_graphs.batch_num_nodes()
            solvent_lens = solvent_graphs.batch_num_nodes()

            try:
                main_pred, aux_pred, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
            except Exception as e:
                print(f"Model forward failed: {e}")
                continue

            loss_main = F.mse_loss(main_pred, labels)
            loss_aux = F.mse_loss(aux_pred, labels)
            loss = loss_main + 0.3 * loss_aux  # weighted auxiliary loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for batch in valid_loader:
                solute_graphs, solvent_graphs, labels = batch
                solute_lens = solute_graphs.batch_num_nodes()
                solvent_lens = solvent_graphs.batch_num_nodes()

                try:
                    main_pred, aux_pred, interaction_map = model([solute_graphs, solvent_graphs, solute_lens, solvent_lens])
                except Exception as e:
                    print(f"Validation forward failed: {e}")
                    continue

                loss_main = F.mse_loss(main_pred, labels)
                loss_aux = F.mse_loss(aux_pred, labels)
                loss = loss_main + 0.3 * loss_aux

                val_losses.append(loss.item())
                val_maes.append(MAE(main_pred, labels).item())

        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_model(model, optimizer, epoch, best_val_loss, project_name)

        scheduler.step()

    print(f"Training complete. Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
    logger.info(f"Training complete. Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
