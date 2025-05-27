import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from generator_model import SMILESGRUGenerator
import re

MODEL_PATH = "smiles_gru_finetuned.pt"

SVR_MODEL_PATH = "../predictions/models/svr/svr_model_average.pkl"
SCALER_PATH = "../predictions/models/svr/svr_scaler_average.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 120
NUM_SAMPLES = 1000
TEMPERATURE = 1.0
OUTPUT_EXCEL = "generated_new.xlsx"

descriptor_cols = [
    "Molecular Weight", "LogP", "TPSA", "Rotatable Bonds", "HBD", "HBA",
    "Aromatic Rings", "Fraction CSP3", "Heavy Atom Count", "Heavy Atom Count (no H)",
    "Molar Refractivity", "Bertz Index", "Balaban Index", "Chi0", "Chi1", "Chiral Centers"
]
df1 = pd.read_excel("../data/average_dataset.xlsx")
df2 = pd.read_excel("predicted_pic50.xlsx")

def prepare_smiles_column(df):
    df["smiles"] = df["smiles"].astype(str).apply(lambda x: re.sub(r"\s+", "", x))
    return df["smiles"].dropna().unique().tolist()
smiles1 = prepare_smiles_column(df1)
smiles2 = prepare_smiles_column(df2)
all_smiles = list(set(smiles1 + smiles2))
vocab = sorted(set("".join(all_smiles))) + ['\n', '~']
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}
def get_descriptors(mol):
    try:
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumHeavyAtoms(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.MolMR(mol),
            Descriptors.BertzCT(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
    except:
        return None

char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(vocab)

model = SMILESGRUGenerator(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

svr_model = joblib.load(SVR_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def sample_smiles(start_token='C', max_length=MAX_LEN, temperature=TEMPERATURE):
    input_char = torch.tensor([[char2idx[start_token]]], dtype=torch.long).to(DEVICE)
    hidden = None
    generated = []

    for _ in range(max_length):
        output, hidden = model(input_char, hidden)
        output = output[:, -1, :] / temperature
        probabilities = torch.softmax(output, dim=-1).squeeze().detach().cpu().numpy()
        next_idx = np.random.choice(len(probabilities), p=probabilities)
        next_char = idx2char[next_idx]
        if next_char == '\n':
            break
        generated.append(next_char)
        input_char = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)

    return ''.join(generated)

results = []
for _ in range(NUM_SAMPLES):
    smi = sample_smiles()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    desc = get_descriptors(mol)
    if desc is None or len(desc) != 16:
        continue

    try:
        desc_scaled = scaler.transform([desc])[0]
        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp)
        full_input = np.concatenate([desc_scaled, fp_array])
        pred = svr_model.predict([full_input])[0]
        results.append({"SMILES": smi, "pIC50": pred})
    except Exception as e:
        continue

if results:
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"{len(df)} молекул збережено у {OUTPUT_EXCEL}")
else:
    print("Жодної валідної молекули не згенеровано")
