import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models import Cigin
from molecular_graph import ConstructMolecularGraph
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNSolVDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        solute = self.data.iloc[idx]['SoluteSMILES']
        solvent = self.data.iloc[idx]['SolventSMILES']
        deltaG = self.data.iloc[idx]['delGsolv']
        return solute, solvent, deltaG

def collate_fn(batch):
    solute_graphs = []
    solvent_graphs = []
    labels = []

    for solute_smiles, solvent_smiles, label in batch:
        sol_graph, sol_feat = ConstructMolecularGraph(solute_smiles)
        solv_graph, solv_feat = ConstructMolecularGraph(solvent_smiles)

        solute_graphs.append((sol_graph, sol_feat))
        solvent_graphs.append((solv_graph, solv_feat))
        labels.append(label)

    return solute_graphs, solvent_graphs, labels

def main():
    # Load and preprocess full dataset
    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    print("Dataset columns:", df.columns)

    # Split: 10% test, then 10% of remaining for validation
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.111, random_state=42)

    train_set = MNSolVDataset(train_df)
    valid_set = MNSolVDataset(valid_df)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = Cigin().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min')

    train(max_epochs=100, model=model, optimizer=optimizer,
          scheduler=scheduler, train_loader=train_loader,
          valid_loader=valid_loader, save_path="best_model.pt")

if __name__ == "__main__":
    main()
