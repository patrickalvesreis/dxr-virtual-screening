"""
SCRIPT: 07_gerar_features_apenas_3D.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Gera dataset APENAS com descritores 3D (RDF, MORSE, WHIM, GETAWAY).
           Remove a parte 2D (ECFP) para reduzir a dimensionalidade.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger

# Suprimir logs
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "descritores_basicos.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "dataset_3d_only.npz")

def gerar_conformero_3d(mol):
    """Gera conformação 3D (ETKDGv3 + MMFF)."""
    mol_3d = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        res = AllChem.EmbedMolecule(mol_3d, params)
        if res == -1:
            res = AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)
            if res == -1: return None
        AllChem.MMFFOptimizeMolecule(mol_3d)
        return mol_3d
    except:
        return None

def get_3d_descriptors(mol_3d):
    """Calcula vetores 3D (RDF, MORSE, WHIM, GETAWAY)."""
    if mol_3d is None: return None

    try:
        # Vetores densos (Float)
        rdf = rdMolDescriptors.CalcRDF(mol_3d)       # ~210 features
        morse = rdMolDescriptors.CalcMORSE(mol_3d)   # ~224 features
        whim = rdMolDescriptors.CalcWHIM(mol_3d)     # ~114 features
        getaway = rdMolDescriptors.CalcGETAWAY(mol_3d) # ~273 features

        # Concatena tudo
        return list(rdf) + list(morse) + list(whim) + list(getaway)
    except Exception as e:
        print(f"Erro ao calcular descritor 3D: {e}")
        return None

def main():
    print("--- GERANDO DATASET: APENAS 3D (GEOMETRIA PURA) ---")

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

        # Gera 3D
        mol_3d = gerar_conformero_3d(mol)

        # Pega APENAS features 3D
        feats_3d = get_3d_descriptors(mol_3d)

        if feats_3d is not None:
            # Checagem de segurança: verificar se tem NaN ou Infinity
            if np.any(np.isnan(feats_3d)) or np.any(np.isinf(feats_3d)):
                print(f"Aviso: ID {mol_id} gerou valores inválidos (NaN/Inf). Pulando.")
                continue

            entry = {
                'ID': mol_id,
                'pIC50': pic50,
                'Features': feats_3d
            }
            data_rows.append(entry)
        else:
            print(f"Aviso: Falha geometria 3D para {mol_id}.")

        if len(data_rows) % 20 == 0:
            print(f"Processados: {len(data_rows)}...")

    print(f"Total Sucesso: {len(data_rows)}")

    # Converter para Numpy
    X = np.array([d['Features'] for d in data_rows], dtype=np.float32)
    y = np.array([d['pIC50'] for d in data_rows], dtype=np.float32)
    ids = np.array([d['ID'] for d in data_rows])

    print(f"Shape Original X: {X.shape}")

    # --- LIMPEZA AUTOMÁTICA (CRUCIAL) ---
    # Remove colunas onde o valor é sempre o mesmo (variância zero)
    # Descritores 3D costumam ter muitas colunas zeradas
    from sklearn.feature_selection import VarianceThreshold
    try:
        selector = VarianceThreshold(threshold=0.0)
        X_clean = selector.fit_transform(X)
        print(f"Shape Após Limpeza (VarianceThreshold): {X_clean.shape}")

        # Salvar
        np.savez_compressed(OUTPUT_FILE, X=X_clean, y=y, ids=ids)
        print(f"Dataset 3D salvo em: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Erro na limpeza de dados: {e}")
        # Salva o original se der erro
        np.savez_compressed(OUTPUT_FILE, X=X, y=y, ids=ids)

if __name__ == "__main__":
    main()
