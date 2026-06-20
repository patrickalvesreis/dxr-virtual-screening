"""
SCRIPT: 07b_gerar_features_apenas_2D.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Gera dataset APENAS com ECFP4 (Morgan Fingerprints).
           Input robusto que não depende de conformação 3D.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "descritores_basicos.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "dataset_2d_only.npz")

def get_ecfp4(mol):
    """Gera Morgan Fingerprint (ECFP4), raio 2, 1024 bits."""
    if mol is None: return None
    # nBits 1024 é padrão ouro para datasets pequenos/médios
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(fp)

def main():
    print("--- GERANDO DATASET: APENAS 2D (ECFP4) ---")

    if not os.path.exists(INPUT_FILE):
        sys.exit("Arquivo básico não encontrado.")

    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['pIC50'])
    print(f"Processando {len(df)} moléculas...")

    data_rows = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        pic50 = row['pIC50']
        mol_id = row['ID']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue

        feats = get_ecfp4(mol)

        if feats is not None:
            entry = {
                'ID': mol_id,
                'pIC50': pic50,
                'Features': feats
            }
            data_rows.append(entry)

    # Converter para Numpy
    X = np.array([d['Features'] for d in data_rows], dtype=np.float32)
    y = np.array([d['pIC50'] for d in data_rows], dtype=np.float32)
    ids = np.array([d['ID'] for d in data_rows])

    print(f"Shape X: {X.shape}")

    # Salvar
    np.savez_compressed(OUTPUT_FILE, X=X, y=y, ids=ids)
    print(f"Dataset 2D salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
