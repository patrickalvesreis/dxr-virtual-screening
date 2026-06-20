"""
SCRIPT: 11_analise_final_gnina.py
DESCRIÇÃO: Lê os SDFs do Gnina, cruza com dados experimentais e gera correlação.
CORREÇÃO: Caminho do CSV ajustado para usar referência relativa.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np

# --- CONFIGURAÇÕES ---
# Pega o diretório onde ESTE script está (Descritores/scripts)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho para os resultados do Docking (ajuste se necessário)
# Assumindo: .../Projeto_DXR.../docking_data/gnina_results
# Vamos tentar subir 3 níveis para chegar na raiz e descer para docking_data
# Estrutura provável:
#   PROJETO/docking_data/gnina_results
#   PROJETO/Banco-de-dados-bindingDB/Descritores/scripts (onde estamos)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
DOCKING_DIR = os.path.join(PROJECT_ROOT, "docking_data", "gnina_results")

# Caminho do CSV experimental (criado pelo script 01)
# Ele está em .../Descritores/outputs/descritores_basicos.csv
INPUT_CSV = os.path.join(BASE_DIR, "..", "outputs", "descritores_basicos.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "..", "outputs", "resultado_FINAL_docking.csv")
PLOT_FILE = os.path.join(BASE_DIR, "..", "outputs", "correlacao_docking_experimental.png")

def parse_sdf_content(content):
    """Extrai scores usando Regex flexível."""
    scores = {}
    m_vina = re.search(r'>\s*<minimizedAffinity>[\s\n]+([-\d\.]+)', content)
    if m_vina: scores['Vina_Score'] = float(m_vina.group(1))

    m_cnn_s = re.search(r'>\s*<CNNscore>[\s\n]+([-\d\.]+)', content)
    if m_cnn_s: scores['CNN_Score'] = float(m_cnn_s.group(1))

    m_cnn_a = re.search(r'>\s*<CNNaffinity>[\s\n]+([-\d\.]+)', content)
    if m_cnn_a: scores['CNN_Affinity'] = float(m_cnn_a.group(1))
    return scores

def main():
    print("--- ANÁLISE FINAL: DOCKING GNINA ---")
    print(f"Buscando CSV experimental em: {INPUT_CSV}")
    print(f"Buscando Docking em: {DOCKING_DIR}")

    if not os.path.exists(INPUT_CSV):
        print("[ERRO] Arquivo CSV experimental ainda não encontrado.")
        print("Verifique se o arquivo 'descritores_basicos.csv' existe na pasta 'outputs' ao lado da pasta 'scripts'.")
        return

    if not os.path.exists(DOCKING_DIR):
        print(f"[ERRO] Pasta de docking não encontrada: {DOCKING_DIR}")
        return

    # 1. Ler Docking
    sdf_files = [f for f in os.listdir(DOCKING_DIR) if f.endswith(".sdf")]
    print(f"Lendo {len(sdf_files)} arquivos SDF...")

    docking_data = []
    for f in sdf_files:
        mol_id = f.replace("gnina_", "").replace(".sdf", "")
        path = os.path.join(DOCKING_DIR, f)
        try:
            with open(path, 'r') as file:
                content = file.read()
            # Pega primeira pose
            best_pose = content.split('$$$$')[0]
            scores = parse_sdf_content(best_pose)
            if scores:
                scores['ID'] = mol_id
                docking_data.append(scores)
        except Exception:
            pass # Ignora erros pontuais de leitura

    df_docking = pd.DataFrame(docking_data)

    # 2. Ler Experimental
    df_exp = pd.read_csv(INPUT_CSV)
    # Converter IDs para string para garantir o casamento perfeito
    df_exp['ID'] = df_exp['ID'].astype(str)

    # 3. Combinar
    df_final = pd.merge(df_exp, df_docking, on="ID", how="inner")
    print(f"Cruzamento realizado: {len(df_final)} moléculas com dados completos.")

    if len(df_final) == 0:
        print("[AVISO] Nenhuma molécula em comum encontrada entre o CSV e os resultados do Docking.")
        print("Verifique se os IDs no CSV (coluna ID) batem com os nomes dos arquivos (gnina_ID.sdf).")
        print(f"Exemplo ID CSV: {df_exp['ID'].iloc[0]}")
        if len(docking_data) > 0:
            print(f"Exemplo ID Docking: {docking_data[0]['ID']}")
        return

    # Salvar
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Tabela final salva em: {OUTPUT_CSV}")

    # 4. Estatística e Gráfico
    df_clean = df_final.dropna(subset=['pIC50', 'CNN_Affinity'])

    if len(df_clean) < 5:
        print("Dados insuficientes para gráfico.")
        return

    y_real = df_clean['pIC50']
    y_cnn = df_clean['CNN_Affinity']

    r2 = r2_score(y_real, y_cnn)
    pearson = y_real.corr(y_cnn)

    print("-" * 30)
    print(f"RESULTADO (R² CNN Affinity): {r2:.4f}")
    print(f"CORRELAÇÃO (Pearson):      {pearson:.4f}")
    print("-" * 30)

    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_real, y=y_cnn, scatter_kws={'alpha':0.6}, color='purple')
    plt.xlabel('pIC50 Experimental')
    plt.ylabel('Predito (CNN Affinity)')
    plt.title(f'Deep Learning Docking: R²={r2:.2f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_FILE)
    print(f"Gráfico gerado: {PLOT_FILE}")

if __name__ == "__main__":
    main()
