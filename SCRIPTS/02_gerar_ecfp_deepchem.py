"""
SCRIPT: 02_gerar_ecfp_deepchem.py
DESCRIÇÃO: Gera Circular Fingerprints (ECFP4) usando DeepChem.
ENTRADA: outputs/descritores_rdkit_completo.csv
SAÍDA: outputs/ecfp_deepchem.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import deepchem as dc

# --- CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "descritores_rdkit_completo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "ecfp_deepchem.csv")

def main():
    print("--- GERANDO FINGERPRINTS (DEEPCHEM) ---")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERRO] Execute o script 01 primeiro para gerar: {INPUT_FILE}")
        sys.exit(1)

    # 1. Carregar dados limpos
    df = pd.read_csv(INPUT_FILE)
    print(f"Carregado: {len(df)} moléculas.")

    # 2. Configurar Featurizer do DeepChem
    # ECFP4 (radius=2), 1024 ou 2048 bits são padrão
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)

    print("Calculando ECFP4 (2048 bits)...")

    # DeepChem featurize espera uma lista de strings (SMILES) ou arquivo
    smiles_list = df["SMILES"].tolist()
    features = featurizer.featurize(smiles_list)

    # features é um array numpy (N_moleculas, 2048)
    print(f"Shape dos fingerprints: {features.shape}")

    # 3. Salvar em CSV
    # Vamos criar colunas Bit_0, Bit_1, etc.
    col_names = [f"Bit_{i}" for i in range(features.shape[1])]
    df_ecfp = pd.DataFrame(features, columns=col_names)

    # Adicionar ID e pIC50 para referência
    df_final = pd.concat([df[["ID", "pIC50"]], df_ecfp], axis=1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"Salvo em: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    main()
