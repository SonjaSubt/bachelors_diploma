import pandas as pd
import joblib
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdmolops, DataStructs
import selfies as sf
from selfies import encoder as smiles_to_selfies, decoder as selfies_to_smiles
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
svr_model = joblib.load("../predictions/models/svr/svr_model_average.pkl")
scaler = joblib.load("../predictions/models/svr/svr_scaler_average.pkl")
df = pd.read_excel("../data/average_dataset.xlsx")

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*16
    try:
        return [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            rdmolops.RemoveHs(mol).GetNumAtoms(),
            Descriptors.MolMR(mol),
            Descriptors.BertzCT(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
    except:
        return [None]*16

def calculate_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

def passes_lipinski(desc):
    if None in desc:
        return False
    mw, logp, _, _, hbd, hba, *_ = desc
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

def mutate_selfies(original_selfie, max_muts=8):
    alphabet = list(sf.get_alphabet_from_selfies([original_selfie]))
    selfie_chars = list(sf.split_selfies(original_selfie))
    if not selfie_chars:
        return original_selfie

    for _ in range(random.randint(1, max_muts)):
        idx = random.randint(0, len(selfie_chars) - 1)
        mutation_type = random.choice(["replace", "insert", "delete"])
        if mutation_type == "replace":
            selfie_chars[idx] = random.choice(alphabet)
        elif mutation_type == "insert":
            selfie_chars.insert(idx, random.choice(alphabet))
        elif mutation_type == "delete" and len(selfie_chars) > 1:
            selfie_chars.pop(idx)

    return "".join(selfie_chars)

def generate_and_score(parents, n_children=1000):
    new_generation = []

    for parent in parents:
        selfie = smiles_to_selfies(parent)
        for _ in range(n_children):
            mutated_selfie = mutate_selfies(selfie)
            mutated_smiles = selfies_to_smiles(mutated_selfie)
            mol = Chem.MolFromSmiles(mutated_smiles)
            if mol is None:
                continue

            desc = calculate_descriptors(mutated_smiles)
            if None in desc:
                continue

            fp = calculate_fingerprint(mutated_smiles)
            if fp is None:
                continue

            desc_scaled = scaler.transform([desc])[0]
            features = np.concatenate([np.array(fp), desc_scaled]).reshape(1, -1)
            pred_pic50 = svr_model.predict(features)[0]

            new_generation.append((mutated_smiles, pred_pic50))

    return new_generation


def run_evolution(seed_smiles_list, generations=5, top_k=50):
    current_parents = seed_smiles_list
    all_results = []

    for gen in range(generations):
        print(f"Покоління {gen+1} — {len(current_parents)} батьків")
        new_candidates = generate_and_score(current_parents)
        if not new_candidates:
            print("Не згенеровано валідних речовин")
            break
        top_offspring = sorted(new_candidates, key=lambda x: -x[1])[:top_k]
        all_results.extend(top_offspring)
        current_parents = [smi for smi, _ in top_offspring]

    return all_results

top_50 = df.sort_values(by="pIC50", ascending=False).head(50)
seed_smiles = top_50["smiles"].tolist()

final_results = run_evolution(seed_smiles, generations=5, top_k=50)

output_df = pd.DataFrame([
    {"generated_smiles": smi, "predicted_pIC50": round(pic, 3)}
    for smi, pic in final_results
]).drop_duplicates(subset="generated_smiles")

output_df.to_excel("generated_evolution.xlsx", index=False)
print(f"Збережено {len(output_df)} молекул до 'generated_evolution.xlsx'")
