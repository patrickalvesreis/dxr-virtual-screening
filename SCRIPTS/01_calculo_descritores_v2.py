"""
SCRIPT: 01_calculo_descritores_v2.py
AUTOR: Patrick / Gemini
DATA: 2025-11-17
DESCRIÇÃO: V2.0 - Calcula TODOS os descritores 2D disponíveis no RDKit (>200).

COMO RODAR (Ambiente chem_gnn - Pixi):
    cd ~/chem_gnn_pixi
    pixi shell
    cd ~/projetos/qsar_gnn_project
    python scripts/01_calculo_descritores.py
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger

# Suprimir avisos do RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURAÇÕES ---
PASTA_DADOS = "/run/timeshift/3126/backup/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/Banco-de-dados-bindingDB/Descritores/Data/"
NOME_ARQUIVO = "smiles_out_Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase.csv"

INPUT_FILE = os.path.join(PASTA_DADOS, NOME_ARQUIVO)

# Caminho de saída
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "descritores_rdkit_completo.csv")

# Colunas do CSV original
COL_SMILES = "canonical_smiles"
COL_ID = "id"
COL_TARGET = "IC50_nM"

def main():
    print("--- INICIANDO PROCESSAMENTO MASSIVO DE DESCRITORES (RDKit) ---")
    print(f"Lendo: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERRO] Arquivo não encontrado.")
        sys.exit(1)

    df_raw = pd.read_csv(INPUT_FILE)

    # 1. Preparar Calculadora de Descritores
    # Pega a lista de todos os nomes de descritores disponíveis no RDKit
    desc_names = [x[0] for x in Descriptors.descList]
    print(f"Calculando {len(desc_names)} descritores para cada molécula...")

    # Cria o objeto calculador (muito mais rápido que chamar um por um)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    valid_data = []
    invalid_count = 0

    # 2. Loop de Processamento
    for index, row in df_raw.iterrows():
        smiles = row[COL_SMILES]
        mol_id = row.get(COL_ID, index)
        target_ic50 = row.get(COL_TARGET, np.nan)

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Adicionar hidrogênios explícitos é bom para alguns descritores 3D/cargas,
            # mas para descritores 2D clássicos do RDKit, o padrão geralmente funciona bem.
            # mol = Chem.AddHs(mol)

            # Calcular TODOS os descritores de uma vez
            desc_values = calculator.CalcDescriptors(mol)

            # Criar dicionário dos descritores
            desc_dict = dict(zip(desc_names, desc_values))

            # Calcular pIC50
            pic50 = np.nan
            if target_ic50 > 0:
                try:
                    pic50 = -np.log10(target_ic50 * 1e-9)
                except:
                    pass

            # Montar linha final
            entry = {
                "ID": mol_id,
                "SMILES": Chem.MolToSmiles(mol), # Canonical
                "IC50_nM": target_ic50,
                "pIC50": pic50
            }
            entry.update(desc_dict)

            valid_data.append(entry)

            # Log de progresso a cada 50 moléculas
            if len(valid_data) % 50 == 0:
                print(f"Processadas: {len(valid_data)}...")

        else:
            invalid_count += 1

    # 3. Salvar
    df_out = pd.DataFrame(valid_data)

    # Limpeza: Remover colunas que deram erro geral (infinito ou NaN)
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Salvar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print("-" * 40)
    print(f"PROCESSAMENTO FINALIZADO")
    print(f"Moléculas válidas: {len(df_out)}")
    print(f"SMILES inválidos: {invalid_count}")
    print(f"Número de descritores gerados: {len(desc_names)}")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print("-" * 40)

if __name__ == "__main__":
    main()
