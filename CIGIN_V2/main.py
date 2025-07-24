# python imports
import pandas as pd
import warnings
import os
import argparse
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

#dgl imports
import dgl

# local imports
from model import CIGINModel
from train import train, evaluate_model, get_metrics
from molecular_graph import get_graph_from_smile
from utils import *
from models.van_GAT import CIGINGAT
from models.van_GGN import CIGINGGN
from models.van_GCN import CIGINGCN
from models.van_GAP import CIGINGAP
from models.van_WAS import CIGINWAS

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin', help="The name of the current project: default: CIGIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                                          "tanh-general", default='dot')
parser.add_argument('--max_epochs', required=False, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=32, help="The batch size for training")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

models = [CIGINGAP(interaction=interaction), CIGINWAS(interaction=interaction)]
model_names = ['cigin_gap', 'cigin_was']

for project_name in model_names:
    if not os.path.isdir("runs/run-" + str(project_name)):
        os.makedirs("./runs/run-" + str(project_name))
        os.makedirs("./runs/run-" + str(project_name) + "/models")


def collate(samples):
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes().tolist())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes().tolist())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # print('solute', self.dataset.iloc[idx]['SoluteSMILES'])
        solute = self.dataset.iloc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)
        # print('solvent',self.dataset.iloc[idx]['SolventSMILES'])
        solvent = self.dataset.iloc[idx]['SolventSMILES']

        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)

        solvent_graph = get_graph_from_smile(solvent)
        delta_g = self.dataset.iloc[idx]['delGsolv']
        return [solute_graph, solvent_graph, [delta_g]]

def main():
    # train_df = pd.read_csv('data/train.csv', sep=";")
    # valid_df = pd.read_csv('data/valid.csv', sep=";")

    df = pd.read_csv('https://raw.githubusercontent.com/adithyamauryakr/CIGIN-DevaLab/refs/heads/master/CIGIN_V2/data/whole_data.csv')
    df.columns = df.columns.str.strip()
    print(df.columns)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.111, random_state=42)

    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)
    test_dataset = Dataclass(test_df)

    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)
    test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=128)
    for model, project_name in zip(models, model_names):
        print('current_model:', project_name)
            
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

        train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)

        # check on testing data:
        model.eval()
        loss, mae_loss = get_metrics(model, test_loader)
        print(f"Model performance on the testing data: Loss: {loss},  MAE_Loss: {mae_loss}")


if __name__ == '__main__':
    main()
