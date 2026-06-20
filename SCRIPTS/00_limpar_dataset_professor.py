# [DIR: ~/chem_gnn_pixi/PROJETO_DXR_ML_GNN/01_Feature_Engineering/scripts]
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

# --- CONFIGURAÇÕES ---
BASE_DIR = "../../"
# Lendo o arquivo ORIGINAL BRUTO (aquele que copiamos primeiro)
INPUT_FILE = os.path.join(BASE_DIR, "00_Dados_Originais", "dataset_dxr_v1.csv")
# Saída: Dataset V2 Limpo
OUTPUT_FILE = os.path.join(BASE_DIR, "00_Dados_Originais", "dataset_dxr_v2_CLEAN.csv")

def limpar_smiles():
    print(f"--- Iniciando Limpeza Rigorosa (Padrão Professor) ---")
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: {INPUT_FILE} não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Total inicial de linhas: {len(df)}")

    # 1. Identificar coluna de SMILES
    col_smiles = None
    for c in ['SMILES', 'canonical_smiles', 'isomeric_smiles', 'Smiles']:
        if c in df.columns:
            col_smiles = c
            break

    if not col_smiles:
        print("ERRO: Coluna de SMILES não encontrada.")
        return

    # Preparar removedor de sais (remove Cl, Na, K, etc.)
    remover = SaltRemover()

    dados_limpos = []
    smiles_unicos = set()
    duplicatas = 0
    inválidos = 0

    for idx, row in df.iterrows():
        smi = row[col_smiles]

        if not isinstance(smi, str):
            inválidos += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            inválidos += 1
            continue

        # A. Remover Contra-íons/Sais
        mol_desalt = remover.StripMol(mol)

        # B. Gerar SMILES Canônico (Isomérico para manter estereoquímica se houver)
        smi_clean = Chem.MolToSmiles(mol_desalt, isomericSmiles=True, canonical=True)

        # C. Verificar Duplicatas
        if smi_clean in smiles_unicos:
            duplicatas += 1
            continue # Pula duplicata

        smiles_unicos.add(smi_clean)

        # Salvar linha limpa
        nova_linha = row.copy()
        nova_linha['SMILES_CLEAN'] = smi_clean
        dados_limpos.append(nova_linha)

    df_final = pd.DataFrame(dados_limpos)

    # Salvar
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"RELATÓRIO DE LIMPEZA:")
    print(f"Inválidos (erro RDKit): {inválidos}")
    print(f"Duplicatas removidas: {duplicatas}")
    print(f"Total Final (V2): {len(df_final)}")
    print("-" * 30)
    print(f"Arquivo limpo salvo em:\n{OUTPUT_FILE}")
    print("Use ESTE arquivo para o Boltz e para gerar descritores daqui para frente!")

if __name__ == "__main__":
    limpar_smiles()
