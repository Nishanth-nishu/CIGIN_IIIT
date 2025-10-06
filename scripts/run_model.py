import torch
from models import Cigin
from molecular_graph import ConstructMolecularGraph

# Sample SMILES strings (you can change these)
solute = "CCO"       # Ethanol
solvent = "O=C=O"    # Carbon dioxide

# Load model
model = Cigin().to("cuda" if torch.cuda.is_available() else "cpu")

# Run model forward pass
with torch.no_grad():
    prediction, interaction_map = model(solute, solvent)

print("Prediction (Solubility):", prediction.item())
print("Interaction Map Shape:", interaction_map.shape)
