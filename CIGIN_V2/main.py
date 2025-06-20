# python imports
import pandas as pd
import warnings
import os
import argparse
import sys
import multiprocessing
import traceback
# rdkit imports
from rdkit import RDLogger, Chem
# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.multiprocessing as mp
# dgl imports
import dgl

# local imports
from model import CIGINModel
from train import train
from molecular_graph import get_graph_from_smile
from utils import get_len_matrix

# Ensure safe multiprocessing on Windows
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass

# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin', help="Project name")
parser.add_argument('--interaction', default='dot',
                    help="Interaction: dot | scaled-dot | general | tanh-general")
parser.add_argument('--max_epochs', default=100, type=int, help="Max epochs")
parser.add_argument('--batch_size', default=4, type=int, help="Batch size")
parser.add_argument('--use_transformer', action='store_true', 
                    help="Use Transformer aggregation instead of Set2Set")
parser.add_argument('--transformer_heads', default=4, type=int,
                    help="Number of attention heads for transformer (default: 4)")
parser.add_argument('--transformer_layers', default=2, type=int,
                    help="Number of transformer layers (default: 2)")
args = parser.parse_args()

project_name = args.name
interaction = args.interaction
max_epochs = args.max_epochs
batch_size = args.batch_size
use_transformer = args.use_transformer
transformer_heads = args.transformer_heads
transformer_layers = args.transformer_layers

# Force CPU for determinism
device = torch.device("cpu")
torch.set_num_threads(2)
print(f"\n📡 Using device: {device}, threads: {torch.get_num_threads()}")

# Create run directories if missing
run_dir = os.path.join("runs", f"run-{project_name}")
os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

def load_data():
    """Load datasets from 'data/train.csv' and 'data/valid.csv'."""
    try:
        train_csv = os.path.join('data', 'train.csv')
        valid_csv = os.path.join('data', 'valid.csv')
        
        if not os.path.exists(train_csv):
            print(f"❌ Error: {train_csv} not found")
            return None, None
        if not os.path.exists(valid_csv):
            print(f"⚠️  Warning: {valid_csv} not found—using train.csv for validation")
            train_df = pd.read_csv(train_csv, sep=';')
            valid_df = train_df.copy()
        else:
            train_df = pd.read_csv(train_csv, sep=';')
            valid_df = pd.read_csv(valid_csv, sep=';')
        
        for col in ['SoluteSMILES', 'SolventSMILES', 'DeltaGsolv']:
            if col not in train_df:
                print(f"❌ Error: Column {col} missing in train.csv")
                return None, None
            if col not in valid_df:
                print(f"❌ Error: Column {col} missing in valid.csv")
                return None, None
        
        print(f"➡️  Loaded datasets: train {len(train_df)} rows, valid {len(valid_df)} rows")
        
        train_dataset = SafeDataset(train_df)
        valid_dataset = SafeDataset(valid_df)
        
        if len(train_dataset) == 0:
            print("❌ Error: No valid training samples found.")
            return None, None
        if len(valid_dataset) == 0:
            print("⚠️  Warning: No valid validation samples found—using training data instead.")
            valid_dataset = train_dataset
        
        return train_dataset, valid_dataset
    
    except Exception as e:
        print("❌ Error loading data:", e)
        traceback.print_exc()
        return None, None

# collate_fn inside your dataset/dataloader
def collate_fn(samples):
    solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels = map(list, zip(*samples))

    # Batch graphs correctly
    batched_solute = dgl.batch(solute_graphs)
    batched_solvent = dgl.batch(solvent_graphs)

    solute_lens = torch.tensor(solute_lens, dtype=torch.float32).unsqueeze(1)
    solvent_lens = torch.tensor(solvent_lens, dtype=torch.float32).unsqueeze(1)

    labels = torch.tensor(labels, dtype=torch.float32)

    return batched_solute, batched_solvent, solute_lens, solvent_lens, labels

class SafeDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.valid_indices = []
        for idx in range(len(self.df)):
            if self._validate_sample(idx):
                self.valid_indices.append(idx)
        print(f"✔️  Valid samples: {len(self.valid_indices)} / {len(self.df)}")
    
    def _validate_sample(self, idx):
        try:
            row = self.df.iloc[idx]
            if pd.isna(row['SoluteSMILES']) or pd.isna(row['SolventSMILES']) or pd.isna(row['DeltaGsolv']):
                return False
            sg = get_graph_from_smile(str(row['SoluteSMILES']))
            vg = get_graph_from_smile(str(row['SolventSMILES']))
            if sg.number_of_nodes()==0 or vg.number_of_nodes()==0:
                return False
            if 'x' not in sg.ndata or sg.ndata['x'].shape[1] != 42:
                return False
            if 'x' not in vg.ndata or vg.ndata['x'].shape[1] != 42:
                return False
            return True
        except:
            return False
    
    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        solute_smiles = row["SoluteSMILES"]
        solvent_smiles = row["SolventSMILES"]
        label = float(row["DeltaGsolv"])
        solute_graph = get_graph_from_smile(solute_smiles)
        solvent_graph = get_graph_from_smile(solvent_smiles)
        solute_len = solute_graph.number_of_nodes()
        solvent_len = solvent_graph.number_of_nodes()
        
        return solute_graph, solvent_graph, solute_len, solvent_len, label



def main():
    print("\n💾 Loading data...")
    train_ds, valid_ds = load_data()
    if train_ds is None:
        return
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    print(f"🧮 Estimated training batches: {len(train_loader)}")
    print(f"🧮 Estimated validation batches: {len(valid_loader)}")

    
    print("\n🧠 Initializing model...")
    print(f"Aggregation method: {'🤖 Transformer' if use_transformer else '📊 Set2Set'}")
    if use_transformer:
        print(f"Transformer config: {transformer_heads} heads, {transformer_layers} layers")
    
    model = CIGINModel(node_input_dim=42, edge_input_dim=10, node_hidden_dim=42,
                       edge_hidden_dim=42, num_step_message_passing=6,
                       interaction=interaction, num_step_set2_set=2,
                       num_layer_set2set=1, use_transformer=use_transformer,
                       transformer_heads=transformer_heads, 
                       transformer_layers=transformer_layers)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print(f"▶️ Starting training: {len(train_ds)} train samples, {len(valid_ds)} valid samples")
    best = train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)
    print(f"\n🏁 Training finished. Best validation loss: {best:.4f}")
    if best is None:
        print("✅ All training epochs completed successfully.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Especially needed for Windows
    main()