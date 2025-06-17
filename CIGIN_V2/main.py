# python imports
import pandas as pd
import warnings
import os
import argparse
import sys
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
args = parser.parse_args()

project_name = args.name
interaction = args.interaction
max_epochs = args.max_epochs
batch_size = args.batch_size

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

def safe_collate(samples):
    """Collate function with validation and fallback."""
    try:
        valid = [s for s in samples if s is not None]
        if not valid:
            raise ValueError("No valid samples to collate")
        
        sol_gs, solv_gs, labels = zip(*valid)
        
        sol_valid, solv_valid, valid_labels = [], [], []
        for i, (sg, vg, lab) in enumerate(zip(sol_gs, solv_gs, labels)):
            try:
                if sg is not None and vg is not None \
                   and 'x' in sg.ndata and 'x' in vg.ndata \
                   and sg.ndata['x'].shape[1] == 42 \
                   and vg.ndata['x'].shape[1] == 42:
                    sol_valid.append(sg)
                    solv_valid.append(vg)
                    valid_labels.append(float(lab))
            except Exception as er:
                print(f"⚠️  Skipping sample {i}: {er}")
        
        if not sol_valid:
            raise ValueError("No valid graph pairs after filtering")
        
        sol_batch = dgl.batch(sol_valid)
        solv_batch = dgl.batch(solv_valid)
        
        sol_sizes = sol_batch.batch_num_nodes().tolist()
        solv_sizes = solv_batch.batch_num_nodes().tolist()
        
        sol_len = get_len_matrix(sol_sizes)
        solv_len = get_len_matrix(solv_sizes)
        
        label_t = torch.tensor(valid_labels, dtype=torch.float32)
        
        return sol_batch, solv_batch, sol_len, solv_len, label_t
    except Exception as e:
        print("❌ Collate error:", e)
        traceback.print_exc()
        raise e

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
        try:
            real_idx = self.valid_indices[idx]
            row = self.df.iloc[real_idx]
            sg = get_graph_from_smile(str(row['SoluteSMILES']))
            vg = get_graph_from_smile(str(row['SolventSMILES']))
            return sg, vg, float(row['DeltaGsolv'])
        except Exception as e:
            print(f"⚠️  Error on __getitem__ idx={idx}: {e}")
            return None

def main():
    print("\n💾 Loading data...")
    train_ds, valid_ds = load_data()
    if train_ds is None:
        return
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=safe_collate, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=safe_collate, num_workers=0, pin_memory=False)
    
    print("\n🧠 Initializing model...")
    model = CIGINModel(node_input_dim=42, edge_input_dim=10, node_hidden_dim=42,
                       edge_hidden_dim=42, num_step_message_passing=6,
                       interaction=interaction, num_step_set2_set=2,
                       num_layer_set2set=1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print(f"▶️ Starting training: {len(train_ds)} train samples, {len(valid_ds)} valid samples")
    best = train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)
    print(f"\n🏁 Training finished. Best validation loss: {best:.4f}")

if __name__ == "__main__":
    main()
