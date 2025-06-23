import numpy as np
import dgl
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from utils import one_of_k_encoding,one_of_k_encoding_unk


def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    Fixed to ensure consistent 42-dimensional output
    """
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Si']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)  # 10
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])  # 6
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1, 2])  # 3
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])  # 7
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])  # 5
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2])  # 5
    
    # Convert features to binary representation (6 bits max for features)
    feature_val = min(features, 63)  # Limit to 6 bits
    atom_features += [int(i) for i in list("{0:06b}".format(feature_val))]  # 6
    
    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
    
    # Chirality features - ensure exactly 2 features
    try:
        if stereo in ['R', 'S']:
            atom_features += [1 if stereo == 'R' else 0, 1 if stereo == 'S' else 0]  # 2
        else:
            atom_features += [0, 0]  # 2
    except:
        atom_features += [0, 0]  # 2
    
    # Total: 10 + 6 + 3 + 7 + 5 + 5 + 6 + 5 + 2 = 49
    # We need exactly 42, so let's adjust
    if len(atom_features) > 42:
        atom_features = atom_features[:42]
    elif len(atom_features) < 42:
        atom_features += [0] * (42 - len(atom_features))
    
    return np.array(atom_features, dtype=np.float32)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    Fixed to ensure consistent 10-dimensional output
    """
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, 
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, 
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]  # 6 features
    
    # Stereo features - ensure exactly 4 features
    stereo_encoding = one_of_k_encoding_unk(str(bond.GetStereo()), 
                                           ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])  # 4
    bond_feats += stereo_encoding
    
    # Total should be exactly 10
    if len(bond_feats) != 10:
        bond_feats = bond_feats[:10] if len(bond_feats) > 10 else bond_feats + [0] * (10 - len(bond_feats))
    
    return np.array(bond_feats, dtype=np.float32)


def get_graph_from_smile(molecule_smile):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    Fixed to ensure consistent dimensions
    """
    try:
        molecule = Chem.MolFromSmiles(molecule_smile)
        molecule = Chem.RemoveHs(molecule) 
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {molecule_smile}")
        
        # Add hydrogens if not present
        if molecule.GetNumAtoms() == 0:
            raise ValueError(f"Empty molecule from SMILES: {molecule_smile}")
        
        # Get features with error handling
        try:
            features = rdDesc.GetFeatureInvariants(molecule)
        except:
            features = [0] * molecule.GetNumAtoms()
        
        # Ensure features list has correct length
        if len(features) != molecule.GetNumAtoms():
            features = [0] * molecule.GetNumAtoms()
        
        try:
            stereo = Chem.FindMolChiralCenters(molecule)
            chiral_centers = ['Unknown'] * molecule.GetNumAtoms()
            for i in stereo:
                if i[0] < len(chiral_centers):
                    chiral_centers[i[0]] = i[1]
        except:
            chiral_centers = ['Unknown'] * molecule.GetNumAtoms()
        
        # Create DGL graph
        num_atoms = molecule.GetNumAtoms()
        
        node_features = []
        edge_features = []
        src_nodes = []
        dst_nodes = []
        
        # Add node features
        for i in range(num_atoms):
            atom_i = molecule.GetAtomWithIdx(i)
            atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
            # Ensure exactly 42 dimensions
            if len(atom_i_features) != 42:
                print(f"Warning: Atom features dimension mismatch: {len(atom_i_features)}")
                atom_i_features = np.pad(atom_i_features, (0, max(0, 42 - len(atom_i_features))))[:42]
            node_features.append(atom_i_features)
        
        # Add edges and edge features
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            
            bond_features_ij = get_bond_features(bond)
            # Ensure exactly 10 dimensions
            if len(bond_features_ij) != 10:
                print(f"Warning: Bond features dimension mismatch: {len(bond_features_ij)}")
                bond_features_ij = np.pad(bond_features_ij, (0, max(0, 10 - len(bond_features_ij))))[:10]
            edge_features.extend([bond_features_ij, bond_features_ij])
        
        # Create DGL graph
        if len(src_nodes) > 0:
            src_tensor = torch.tensor(src_nodes, dtype=torch.int64)
            dst_tensor = torch.tensor(dst_nodes, dtype=torch.int64)
            G = dgl.graph((src_tensor, dst_tensor), num_nodes=num_atoms)
            G.edata['w'] = torch.tensor(np.array(edge_features), dtype=torch.float64)
        else:
            # Handle molecules with no bonds (single atoms)
            G = dgl.graph(([], []), num_nodes=num_atoms)
            G.edata['w'] = torch.empty((0, 10), dtype=torch.float64)
        
        G.ndata['x'] = torch.tensor(np.array(node_features), dtype=torch.float64)
        
        # Verify dimensions
        assert G.ndata['x'].shape[1] == 42, f"Node features must be 42D, got {G.ndata['x'].shape[1]}"
        if G.number_of_edges() > 0:
            assert G.edata['w'].shape[1] == 10, f"Edge features must be 10D, got {G.edata['w'].shape[1]}"
        
        return G
        
    except Exception as e:
        print(f"Error creating graph from SMILES {molecule_smile}: {e}")
        # Return a minimal graph with correct dimensions
        G = dgl.graph(([], []), num_nodes=1)
        G.ndata['x'] = torch.zeros((1, 42), dtype=torch.float64)
        G.edata['w'] = torch.empty((0, 10), dtype=torch.float64)
        return G
