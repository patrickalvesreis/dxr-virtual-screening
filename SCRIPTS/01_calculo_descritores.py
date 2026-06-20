import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Suprimir avisos do RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURAÇÕES ---
# Caminho ABSOLUTO da pasta de dados (conforme você pediu)
PASTA_DADOS = "/run/timeshift/3126/backup/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/Banco-de-dados-bindingDB/Descritores/Data/"
NOME_ARQUIVO = "smiles_out_Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase.csv"

# Monta o caminho do input
INPUT_FILE = os.path.join(PASTA_DADOS, NOME_ARQUIVO)

# Caminho de saída: salva em ../outputs em relação a onde o script está
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Pasta onde este script está
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "descritores_basicos.csv")

# Colunas do CSV original
COL_SMILES = "canonical_smiles"
COL_ID = "id"
COL_TARGET = "IC50_nM"

def calcular_descritores(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "QED": Descriptors.qed(mol)
    }

def main():
    print("--- INICIANDO PROCESSAMENTO ---")
    print(f"Lendo: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERRO] Arquivo não encontrado: {INPUT_FILE}")
        sys.exit(1)

    try:
        df_raw = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV: {e}")
        sys.exit(1)

    if COL_SMILES not in df_raw.columns:
        print(f"[ERRO] Coluna '{COL_SMILES}' não encontrada.")
        sys.exit(1)

    valid_data = []
    invalid_count = 0

    # Iterar sobre as moléculas
    for index, row in df_raw.iterrows():
        smiles = row[COL_SMILES]
        mol_id = row.get(COL_ID, index)
        target_ic50 = row.get(COL_TARGET, np.nan)

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Calcular pIC50
            pic50 = np.nan
            if target_ic50 > 0:
                try:
                    pic50 = -np.log10(target_ic50 * 1e-9)
                except:
                    pic50 = np.nan

            # Calcular descritores RDKit
            desc = calcular_descritores(mol)

            # Montar dicionário (AQUI ESTAVA O ERRO ANTES)
            entry = {
                "ID": mol_id,
                "SMILES": Chem.MolToSmiles(mol),
                "IC50_nM": target_ic50,
                "pIC50": pic50
            }
            # Adiciona os descritores ao dicionário
            entry.update(desc)

            valid_data.append(entry)
        else:
            invalid_count += 1

    # Salvar resultados
    df_out = pd.DataFrame(valid_data)
    df_clean = df_out.dropna(subset=["pIC50"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"Processamento Finalizado!")
    print(f"Total processado: {len(df_out)}")
    print(f"Salvo em: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    main()
