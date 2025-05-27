import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

df = pd.read_excel("generated_new.xlsx")
valid_smiles = df["SMILES"].dropna().tolist()
pic50 = df["pIC50"]

total = 1000
valid = len(valid_smiles)
valid_percent = (valid / total) * 100

mean_pic50 = pic50.mean()
median_pic50 = pic50.median()
above_6 = (pic50 > 6).sum()
above_7 = (pic50 > 7).sum()

unique_smiles = len(set(valid_smiles))
unique_ratio = unique_smiles / valid

fps = []
for smi in valid_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fps.append(fp)

def tanimoto_diversity(fps):
    n = len(fps)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    avg_sim = np.mean(sims)
    return 1 - avg_sim  

diversity = tanimoto_diversity(fps)

print(f"Валідні молекули: {valid} / {total} ({valid_percent:.2f}%)")
print(f"Унікальні SMILES: {unique_smiles} ({unique_ratio:.2f} від валідних)")
print(f"Середнє pIC50: {mean_pic50:.2f}")
print(f"Медіанне pIC50: {median_pic50:.2f}")
print(f"pIC50 > 6: {above_6} ({(above_6/valid)*100:.1f}%)")
print(f"pIC50 > 7: {above_7} ({(above_7/valid)*100:.1f}%)")
print(f"Різноманітність (1 - середнє Tanimoto): {diversity:.3f}")
