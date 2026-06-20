"""
SCRIPT: 07_preparar_ligantes.py
AMBIENTE: REQUER 'micromamba activate docking'
DESCRIÇÃO: Lê SMILES do CSV, gera conformação 3D, protona e converte para PDBQT (Meeko).
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTMolecule

# --- CONFIGURAÇÕES ---
# Caminho base do projeto
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"

# Onde está o seu CSV com os SMILES? (Ajuste se necessário)
# Estou assumindo que você copiou o arquivo original para uma pasta 'data' ou manteve na pasta original
# Se estiver no backup do timeshift, aponte para lá. Vou usar um caminho sugerido dentro do projeto:
INPUT_CSV = os.path.join(PROJECT_DIR, "Banco-de-dados-bindingDB/Descritores/Data/smiles_out_Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase.csv")

# Onde salvar os PDBQTs dos ligantes
OUTPUT_DIR = os.path.join(PROJECT_DIR, "docking_data", "ligands_pdbqt")

def main():
    print("--- PREPARAÇÃO DE LIGANTES (SMILES -> PDBQT) ---")

    # Verificar entrada
    if not os.path.exists(INPUT_CSV):
        print(f"[ERRO] Arquivo CSV não encontrado em: {INPUT_CSV}")
        print("Por favor, edite a variável INPUT_CSV no script com o caminho correto.")
        return

    # Criar pasta de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Salvando ligantes em: {OUTPUT_DIR}")

    # Ler dados
    df = pd.read_csv(INPUT_CSV)

    # Ajustar nome da coluna de SMILES se necessário (confirmei pelo seu upload anterior)
    col_smiles = 'canonical_smiles'
    col_id = 'id'

    if col_smiles not in df.columns:
        print(f"[ERRO] Coluna '{col_smiles}' não encontrada. Colunas disponíveis: {df.columns}")
        return

    print(f"Total de moléculas para processar: {len(df)}")

    sucesso = 0
    falhas = 0

    for idx, row in df.iterrows():
        mol_id = str(row.get(col_id, idx)) # Usa ID ou índice
        smiles = row[col_smiles]

        # 1. Criar Molécula RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[AVISO] SMILES inválido ID {mol_id}")
            falhas += 1
            continue

        # 2. Preparar 3D
        try:
            # Adicionar Hidrogênios (Crucial para pH fisiológico e 3D)
            mol = Chem.AddHs(mol)

            # Gerar coordenadas 3D (ETKDGv3 é o melhor algoritmo atual do RDKit)
            params = AllChem.ETKDGv3()
            params.useSmallRingTorsions = True # Ajuda em anéis difíceis
            res = AllChem.EmbedMolecule(mol, params)

            # Se falhar, tenta aleatório
            if res == -1:
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if res == -1: raise ValueError("Falha no Embedding 3D")

            # Otimização de energia rápida (MMFF94) para relaxar a estrutura
            AllChem.MMFFOptimizeMolecule(mol)

        except Exception as e:
            print(f"[ERRO 3D] ID {mol_id}: {e}")
            falhas += 1
            continue

        # 3. Converter para PDBQT com Meeko
        try:
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            # Retorna a string do arquivo PDBQT
            pdbqt_string = preparator.write_pdbqt_string()

            # Salvar Arquivo
            filename = os.path.join(OUTPUT_DIR, f"ligand_{mol_id}.pdbqt")
            with open(filename, 'w') as f:
                f.write(pdbqt_string)

            sucesso += 1
            if sucesso % 50 == 0:
                print(f"Processados: {sucesso}...")

        except Exception as e:
            print(f"[ERRO MEEKO] ID {mol_id}: {e}")
            falhas += 1

    print("-" * 30)
    print(f"Processamento Concluído.")
    print(f"Sucesso: {sucesso}")
    print(f"Falhas:  {falhas}")

    if sucesso > 0:
        print("\nPróximo passo: Executar o Vina!")

if __name__ == "__main__":
    main()
