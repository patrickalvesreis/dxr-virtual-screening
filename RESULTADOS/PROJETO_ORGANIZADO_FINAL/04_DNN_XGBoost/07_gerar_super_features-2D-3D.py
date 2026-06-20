"""
SCRIPT: 07_gerar_super_features-2D-3D.py
DESCRIÇÃO: Gera um dataset com ECFP2+4+6 concatenados E descritores 3D (RDF, MORSE, WHIM).
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit import RDLogger

# Suprimir logs de warnings do RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "descritores_basicos.csv") # Usando o CSV limpo básico
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "dataset_super_features.csv")

def gerar_conformero_3d(mol):
    """
    Tenta gerar uma conformação 3D para a molécula.
    Retorna a molécula com H adicionados e geometria otimizada.
    """
    mol_3d = Chem.AddHs(mol)
    try:
        # ETKDGv3 é o estado da arte no RDKit para conformeros
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        res = AllChem.EmbedMolecule(mol_3d, params)

        if res == -1: # Falha
             # Tenta parametro mais relaxado
             res = AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)
             if res == -1: return None

        # Otimização rápida de campo de força (MMFF94)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        return mol_3d
    except:
        return None

def get_ecfps_concatenated(mol):
    """Gera ECFP 2, 4 e 6 e concatena."""
    # Usando nBits menor para cada um para não explodir a memória (512 * 3 = 1536 bits)
    # Se quiser mais poder, aumente para 1024
    nbits = 1024
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=nbits) # Radius 1
    fp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits) # Radius 2
    fp6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits) # Radius 3

    # Converter para lista
    vec = list(fp2) + list(fp4) + list(fp6)
    return vec

def get_3d_descriptors(mol_3d):
    """Calcula vetores de descritores 3D complexos."""
    if mol_3d is None:
        # Retorna lista de zeros se falhou no 3D
        # Precisamos saber o tamanho exato dos vetores...
        # RDF=210, MORSE=224, WHIM=114, GETAWAY=273 (aproximadamente, varia com config)
        # Para evitar erro de dimensão, vamos calcular num dummy primeiro na main ou tratar exceção.
        return None

    # 1. RDF (Radial Distribution Function)
    rdf = rdMolDescriptors.CalcRDF(mol_3d)

    # 2. MORSE (3D Molecule Representation of Structures based on Electron diffraction)
    morse = rdMolDescriptors.CalcMORSE(mol_3d)

    # 3. WHIM (Weighted Holistic Invariant Molecular descriptors)
    whim = rdMolDescriptors.CalcWHIM(mol_3d)

    # 4. GETAWAY (Geometry, Topology, and Atom-Weights Assembly)
    getaway = rdMolDescriptors.CalcGETAWAY(mol_3d)

    return list(rdf) + list(morse) + list(whim) + list(getaway)

def main():
    print("--- GERANDO SUPER FEATURES (2D + 3D) ---")

    if not os.path.exists(INPUT_FILE):
        sys.exit("Arquivo básico não encontrado.")

    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['pIC50'])
    print(f"Processando {len(df)} moléculas...")

    data_rows = []

    # Para saber o tamanho do vetor 3D em caso de falha
    dim_3d = 0

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        pic50 = row['pIC50']
        mol_id = row['ID']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue

        # 1. Calcular 2D (Sempre funciona)
        feats_2d = get_ecfps_concatenated(mol)

        # 2. Calcular 3D
        mol_3d = gerar_conformero_3d(mol)
        feats_3d = get_3d_descriptors(mol_3d)

        if feats_3d is not None:
            dim_3d = len(feats_3d)
            # Concatenar tudo
            full_vector = feats_2d + feats_3d

            entry = {
                'ID': mol_id,
                'pIC50': pic50,
                'Features': full_vector
            }
            data_rows.append(entry)
        else:
            print(f"Aviso: Falha ao gerar 3D para {mol_id}. Pulando.")

        if len(data_rows) % 20 == 0:
            print(f"Processados: {len(data_rows)}...")

    print(f"Total Sucesso: {len(data_rows)}")

    # Transformar em DataFrame expandido é pesado.
    # Vamos salvar como um arquivo Python Pickle (.pkl) ou Numpy (.npy) comprimido
    # pois CSV com 5000 colunas fica lento.

    # Preparar matriz X e vetor y
    X = np.array([d['Features'] for d in data_rows], dtype=np.float32)
    y = np.array([d['pIC50'] for d in data_rows], dtype=np.float32)
    ids = np.array([d['ID'] for d in data_rows])

    print(f"Shape Final X: {X.shape}") # (N_samples, ~4000)
    print(f"Shape Final y: {y.shape}")

    output_npz = os.path.join(BASE_DIR, "..", "outputs", "super_dataset.npz")
    np.savez_compressed(output_npz, X=X, y=y, ids=ids)

    print(f"Dataset comprimido salvo em: {output_npz}")

if __name__ == "__main__":
    main()
