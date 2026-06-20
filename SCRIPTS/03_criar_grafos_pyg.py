"""
SCRIPT: 03_criar_grafos_pyg.py
DESCRIÇÃO: Converte SMILES em objetos PyTorch Geometric (Data).
SAÍDA: outputs/grafos_dxr.pt (Arquivo binário do PyTorch)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops

# --- CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "descritores_rdkit_completo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "grafos_dxr.pt")

def one_hot_encoding(x, permitted_list):
    """Utilitário para One-Hot Encoding seguro."""
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == possible) for possible in permitted_list]
    return binary_encoding

def get_atom_features(atom):
    """
    Gera features para um átomo:
    1. Tipo Atômico (C, N, O, F, P, S, Cl, Br, I, Outro)
    2. Grau (Degree)
    3. Hibridização
    4. Aromaticidade
    """
    # Lista de átomos permitidos no nosso dataset (ajuste se necessário)
    permitted_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Unknown']
    atom_type = one_hot_encoding(atom.GetSymbol(), permitted_atoms)

    # Grau do átomo
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

    # Hibridização
    hybridization = one_hot_encoding(
        str(atom.GetHybridization()),
        ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )

    # Aromaticidade (booleano convertido para int)
    is_aromatic = [int(atom.GetIsAromatic())]

    return atom_type + degree + hybridization + is_aromatic

def smiles_to_graph(smiles, y_val, mol_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # 1. Node Features (X)
    xs = []
    for atom in mol.GetAtoms():
        xs.append(get_atom_features(atom))
    x = torch.tensor(xs, dtype=torch.float)

    # 2. Edge Index (Conectividade)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Grafo não-direcionado: adiciona (i,j) e (j,i)
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if not edge_indices: # Caso seja um íon solto sem ligações
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # 3. Target (y)
    y = torch.tensor([y_val], dtype=torch.float)

    # Criar objeto Data
    data = Data(x=x, edge_index=edge_index, y=y, mol_id=mol_id)
    return data

def main():
    print("--- GERANDO GRAFOS PYTORCH GEOMETRIC ---")

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"Arquivo {INPUT_FILE} não encontrado.")

    df = pd.read_csv(INPUT_FILE)

    data_list = []
    print(f"Processando {len(df)} moléculas...")

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        target = row['pIC50']
        mid = row['ID']

        graph = smiles_to_graph(smiles, target, mid)
        if graph is not None:
            data_list.append(graph)

    print(f"Grafos criados com sucesso: {len(data_list)}")

    # Salvar lista de grafos
    torch.save(data_list, OUTPUT_FILE)
    print(f"Dataset salvo em: {OUTPUT_FILE}")

    # Check rápido
    if len(data_list) > 0:
        print("\nExemplo de grafo (primeiro):")
        print(data_list[0])
        print(f"Node features shape: {data_list[0].x.shape} (Átomos, Features)")
        print(f"Edges shape: {data_list[0].edge_index.shape}")

if __name__ == "__main__":
    main()
