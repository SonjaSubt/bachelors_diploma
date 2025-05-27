import pandas as pd
import os

df = pd.read_excel("dataset.xlsx")

group_column = "smiles"

best_df = df.loc[df.groupby(group_column)["pIC50"].idxmax()].reset_index(drop=True)
os.makedirs("data", exist_ok=True)
best_df.to_excel("data/best_dataset.xlsx", index=False)

average_df = df.groupby(group_column).agg({
    'pIC50': 'mean',
    'source': lambda x: ', '.join(sorted(set(x))),
    'morgan_fingerprint': 'first',
    'Molecular Weight': 'mean',
    'LogP': 'mean',
    'TPSA': 'mean',
    'Rotatable Bonds': 'mean',
    'HBD': 'mean',
    'HBA': 'mean',
    'Aromatic Rings': 'mean',
    'Fraction CSP3': 'mean',
    'Heavy Atom Count': 'mean',
    'Heavy Atom Count (no H)': 'mean',
    'Molar Refractivity': 'mean',
    'Bertz Index': 'mean',
    'Balaban Index': 'mean',
    'Chi0': 'mean',
    'Chi1': 'mean',
    'Chiral Centers': 'mean'
}).reset_index()

average_df.to_excel("data/average_dataset.xlsx", index=False)

print("Файли збережено")
