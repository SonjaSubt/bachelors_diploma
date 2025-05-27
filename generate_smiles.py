import torch
from generation.generator_model import SMILESGRUGenerator
from generator_model import SMILESLSTMGenerator
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SanitizeMol
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Параметри
MIN_LEN = 10
MAX_LEN = 120
NUM_SAMPLES = 500
TEMPERATURE = 0.6

# Словник без PAD_CHAR
vocab = sorted([
    '/', '7', '#', '4', '@', '+', 'N', '[', '\\', '=', '3', 'C', 'o', 'l',
    'c', '2', ']', 'r', ')', '.', 'O', 'S', '5', 'i', 'F', 'I', 'H', '-',
    'n', '6', 'B', '(', '1', '\n', '~'
])
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}

# Завантаження моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SMILESGRUGenerator(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load("smiles_gru_finetuned.pt", map_location=device))
model.eval()

def sample(preds, temperature=1.0):
    preds = torch.softmax(preds / temperature, dim=-1)
    return torch.multinomial(preds, 1).item()

def generate_smiles(start_token='\n'):
    input_seq = torch.tensor([char2idx[start_token]], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    result = []

    for _ in range(MAX_LEN):
        output, hidden = model(input_seq, hidden)
        next_char_logits = output[0, -1]
        next_idx = sample(next_char_logits, temperature=TEMPERATURE)
        next_char = idx2char[next_idx]

        if next_char == '\n':
            break

        result.append(next_char)
        input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return ''.join(result)

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        SanitizeMol(mol)
        return True
    except:
        return False

generated = []
print("Генерація SMILES...")
for i in range(NUM_SAMPLES):
    smiles = generate_smiles().replace('~', '')
    if i < 50:
        print(f"[{i+1}] {smiles}")
    try:
        if is_valid_smiles(smiles) and len(smiles) >= MIN_LEN:
            generated.append(smiles)
    except:
        continue

if generated:
    df = pd.DataFrame(generated, columns=["generated_smiles"])
    df.to_excel("generated_smiles.xlsx", index=False)
    print(f"\nЗгенеровано {len(generated)} валідних SMILES та збережено у 'generated_smiles.xlsx'")
else:
    print("\n Жоден згенерований SMILES не є валідним. Файл не створено.")

#print("Валідні SMILES:", generated)
#print("Кількість валідних:", len(generated))
