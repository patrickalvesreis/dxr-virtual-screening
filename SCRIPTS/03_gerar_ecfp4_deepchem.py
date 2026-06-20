# [DIR: ~/chem_gnn_pixi/PROJETO_DXR_ML_GNN/01_Feature_Engineering/scripts]
import os
import pandas as pd
import deepchem as dc
import numpy as np

# --- CONFIGURAÇÕES ---
BASE_DIR = "../../"
# Lemos do arquivo já processado para garantir que temos os mesmos pIC50
INPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "descritores_rdkit_completo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "ecfp4_features.csv")

def gerar_fingerprints():
    print("--- Gerando ECFP4 com DeepChem ---")
    if not os.path.exists(INPUT_FILE):
        print("ERRO: Arquivo de entrada não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)
    smiles_list = df['SMILES'].tolist()
    ids = df['ID'].tolist()
    pic50s = df['pIC50'].tolist()

    print(f"Total de moléculas: {len(smiles_list)}")

    # Configurar Featurizer (ECFP4, raio 2, 1024 bits)
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)

    print("Calculando features... (pode demorar alguns segundos)")
    features = featurizer.featurize(smiles_list)

    # Verificar falhas
    valid_idxs = []
    for i, feat in enumerate(features):
        if feat.size > 0: # DeepChem retorna array vazio se falhar
            valid_idxs.append(i)

    print(f"Moléculas com sucesso: {len(valid_idxs)} / {len(smiles_list)}")

    # Montar DataFrame final
    # Colunas de bits: bit_0, bit_1, ... bit_1023
    col_names = [f"bit_{i}" for i in range(1024)]

    # Filtrar apenas os válidos
    features_valid = features[valid_idxs]
    ids_valid = [ids[i] for i in valid_idxs]
    pic50_valid = [pic50s[i] for i in valid_idxs]

    df_ecfp = pd.DataFrame(features_valid, columns=col_names)
    df_ecfp.insert(0, "ID", ids_valid)
    df_ecfp.insert(1, "pIC50", pic50_valid)

    # Salvar
    df_ecfp.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset ECFP4 salvo em: {OUTPUT_FILE}")
    print(f"Shape final: {df_ecfp.shape}")

if __name__ == "__main__":
    gerar_fingerprints()
