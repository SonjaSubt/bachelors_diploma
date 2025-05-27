import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdmolops, AllChem
import numpy as np

input_file = "dataset_lopez+pubchem.xlsx"
output_file = "dataset.xlsx"

df = pd.read_excel(input_file)
smiles_list = df['smiles'].tolist()

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 17

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    sp3_fraction = Descriptors.FractionCSP3(mol)
    heavy_atom_count = Descriptors.HeavyAtomCount(mol)
    heavy_atom_count_no_h = rdmolops.RemoveHs(mol).GetNumAtoms()
    molar_refractivity = Descriptors.MolMR(mol)
    bertz_index = Descriptors.BertzCT(mol)
    balaban_index = Descriptors.BalabanJ(mol)
    chi_0 = Descriptors.Chi0(mol)
    chi_1 = Descriptors.Chi1(mol)

    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral_centers = len(chiral_centers)

    return [mw, logp, tpsa, rot_bonds, hbd, hba, aromatic_rings,
            sp3_fraction, heavy_atom_count, heavy_atom_count_no_h,
            molar_refractivity, bertz_index, balaban_index, chi_0, chi_1, num_chiral_centers]

def calculate_fingerprint_string(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return ''.join([str(int(bit)) for bit in fp])

# Обчислення
descriptors_list = []
fingerprint_str_list = []

for smiles in smiles_list:
    descriptors = calculate_descriptors(smiles)
    fingerprint = calculate_fingerprint_string(smiles)
    descriptors_list.append(descriptors)
    fingerprint_str_list.append(fingerprint)

# Датафрейми
descriptors_df = pd.DataFrame(descriptors_list, columns=[
    'Molecular Weight', 'LogP', 'TPSA', 'Rotatable Bonds', 'HBD', 'HBA', 'Aromatic Rings',
    'Fraction CSP3', 'Heavy Atom Count', 'Heavy Atom Count (no H)',
    'Molar Refractivity', 'Bertz Index', 'Balaban Index', 'Chi0', 'Chi1', 'Chiral Centers'
])

df['Fingerprint'] = fingerprint_str_list

# Об'єднання і збереження
df = pd.concat([df, descriptors_df], axis=1)
df.to_excel(output_file, index=False)

print(f"Descriptors and fingerprint strings saved to {output_file}")