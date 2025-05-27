import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText

file_path = "diploma_data2/dataset_lopez+pubchem.xlsx"
df = pd.read_excel(file_path)

print(df.columns)
if 'smiles' not in df.columns:
    raise ValueError("no columns")

def compute_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_str = BitVectToText(fp)

        return fp_str
    except:
        return None

df[['morgan_fingerprint']] = df.apply(
    lambda row: compute_fingerprint(row['smiles']), axis=1, result_type='expand'
)

output_file = 'diploma_data2/dataset_final_finger.xlsx'
df.to_excel(output_file, index=False)
print(f"Saved in: {output_file}")
