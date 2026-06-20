"""
SCRIPT: 03_criar_grafos_hibridos.py
DESCRIÇÃO: Gera grafos PyG enriquecidos com features globais (RDKit Descriptors).
SAÍDA: outputs/grafos_hibridos_dxr.pt
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

# --- CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DESC = os.path.join(BASE_DIR, "..", "outputs", "descritores_rdkit_completo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "grafos_hibridos_dxr.pt")

# Features atômicas (Expandidas)
def one_hot(x, permitted):
    if x not in permitted: x = permitted[-1]
    return [int(x == s) for s in permitted]

def get_atom_features(atom):
    # 1. Simbolo
    atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Unknown']
    feats = one_hot(atom.GetSymbol(), atoms)

    # 2. Grau e Hibridização
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feats += one_hot(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    # 3. Propriedades Químicas Extras
    feats += [int(atom.GetIsAromatic())]
    feats += [int(atom.GetFormalCharge())]
    feats += [int(atom.IsInRing())]

    # 4. Massa (normalizada grosseiramente por 100)
    feats += [atom.GetMass() * 0.01]

    return feats

def main():
    print("--- GERANDO GRAFOS HÍBRIDOS (EXTRA FEATURES) ---")

    if not os.path.exists(INPUT_DESC):
        sys.exit(f"Arquivo {INPUT_DESC} não encontrado.")

    # 1. Ler Dataset e Descritores
    df = pd.read_csv(INPUT_DESC)
    df = df.dropna(subset=['pIC50']) # Garantir target

    print(f"Dataset base: {len(df)} moléculas")

    # 2. Preparar 'Global Features' (Descritores RDKit)
    # Vamos pegar colunas numéricas, exceto IDs e Targets
    ignore_cols = ['ID', 'SMILES', 'IC50_nM', 'pIC50']
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    # Limpeza de NaNs/Infinitos nos descritores
    features_raw = df[feat_cols].values
    features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalização (StandardScaler) - CRUCIAL para convergir
    print("Normalizando descritores globais...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)

    # Converter para tensor
    global_features_tensor = torch.tensor(features_scaled, dtype=torch.float)

    data_list = []

    # 3. Loop de Criação
    for i, (idx, row) in enumerate(df.iterrows()):
        smiles = row['SMILES']
        target = row['pIC50']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue

        # Node Features (x)
        xs = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(xs, dtype=torch.float)

        # Edge Index
        edge_indices = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.append([u, v])
            edge_indices.append([v, u])

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Global Features (vetor extra para essa molécula)
        # Pegamos a linha i correspondente
        g_feat = global_features_tensor[i].unsqueeze(0) # Shape [1, N_descritores]

        y = torch.tensor([target], dtype=torch.float)

        # Objeto Data Enriquecido
        data = Data(x=x, edge_index=edge_index, y=y, global_x=g_feat)
        data_list.append(data)

    print(f"Grafos gerados: {len(data_list)}")
    print(f"Dimensão Node Features: {data_list[0].x.shape[1]}")
    print(f"Dimensão Global Features: {data_list[0].global_x.shape[1]}")

    torch.save(data_list, OUTPUT_FILE)
    print(f"Salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
