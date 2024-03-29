
import dgl
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint
from torch.utils.data import Dataset
def mol_to_graph(mol):
    if mol is None:
        return None
    num_atoms = mol.GetNumAtoms()
    g = dgl.graph(([], []))
    g.add_nodes(num_atoms)

    bond_types = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_types.extend([bond_type, bond_type])
        g.add_edges([start, end], [end, start])

    h_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    g.ndata['h'] = torch.tensor(h_feats).unsqueeze(1).float()
    g.edata['e'] = torch.tensor(bond_types).unsqueeze(1).float()

    return g


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph(mol)

        if graph is None:
            return None, None, None

        # Extract Molecular Descriptors and Fingerprints
        descriptor_names = [
            "MolWt",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "MolLogP",
            "NumRotatableBonds",
        ]
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
        fingerprints = [
            int(bit) for fp in [
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512).ToBitString(),
                MACCSkeys.GenMACCSKeys(mol).ToBitString(),
                RDKFingerprint(mol, fpSize=512).ToBitString()
            ] for bit in fp
        ]
        fingerprint_names = [f"Fingerprint_{i}" for i in range(len(fingerprints))]

        feature_names = descriptor_names + fingerprint_names

        #print("Feature names:")
        #print(feature_names)

        features = torch.tensor(descriptors + fingerprints).float()
        label = self.labels[idx]

        return graph, features, label


def collate(samples):
    valid_samples = [s for s in samples if s[0] is not None]
    graphs, features, labels = map(list, zip(*valid_samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(features), torch.tensor(labels)