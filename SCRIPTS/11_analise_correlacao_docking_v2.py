"""
SCRIPT: 11_analise_correlacao_docking_v2.py
AMBIENTE: 'micromamba activate docking' ou 'chem_gnn'
DESCRIÇÃO: Cruza os resultados do Docking com o pIC50 experimental.
           CORREÇÃO: Uso de caminhos absolutos para achar o arquivo CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr

# --- CONFIGURAÇÕES ---
# Caminho Raiz do Projeto (O mesmo do Script 09)
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"

# Caminhos Absolutos
DOCKING_CSV = os.path.join(PROJECT_DIR, "outputs", "resultados_docking_gnina.csv")

# O arquivo experimental (descritores_basicos) costuma estar na mesma pasta outputs raiz ou na descritores/outputs
# Vamos tentar primeiro na raiz, se falhar tentamos o relativo
EXP_CSV_1 = os.path.join(PROJECT_DIR, "outputs", "descritores_basicos.csv")
EXP_CSV_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "descritores_basicos.csv")

OUTPUT_PLOT = os.path.join(PROJECT_DIR, "outputs", "correlacao_docking_vs_pIC50.png")
FINAL_TABLE = os.path.join(PROJECT_DIR, "outputs", "tabela_final_ranking_unificado.csv")

def main():
    print("--- ANÁLISE FINAL: DOCKING vs EXPERIMENTAL ---")

    # 1. Carregar Dados do Docking
    if os.path.exists(DOCKING_CSV):
        print(f"Carregando Docking de: {DOCKING_CSV}")
        df_dock = pd.read_csv(DOCKING_CSV)
    else:
        print(f"ERRO FATAL: Não achei o arquivo {DOCKING_CSV}")
        print("Verifique se o Script 09 rodou até o final e gerou o CSV.")
        return

    # 2. Carregar Dados Experimentais
    if os.path.exists(EXP_CSV_1):
        print(f"Carregando Experimental de: {EXP_CSV_1}")
        df_exp = pd.read_csv(EXP_CSV_1)
    elif os.path.exists(EXP_CSV_2):
        print(f"Carregando Experimental de: {EXP_CSV_2}")
        df_exp = pd.read_csv(EXP_CSV_2)
    else:
        print("ERRO FATAL: Não achei o arquivo descritores_basicos.csv em lugar nenhum.")
        return

    # Converter ID para string para garantir o merge correto
    df_dock['ID'] = df_dock['ID'].astype(str)
    df_exp['ID'] = df_exp['ID'].astype(str)

    # 3. Merge (Juntar tabelas)
    df_full = pd.merge(df_exp[['ID', 'pIC50', 'SMILES']],
                       df_dock[['ID', 'minimizedAffinity', 'CNNscore', 'CNNaffinity']],
                       on='ID', how='inner')

    print(f"Total de moléculas cruzadas (Docking + pIC50): {len(df_full)}")

    if len(df_full) < 5:
        print("ALERTA: Pouquíssimos dados cruzados. Verifique se os IDs do docking (nomes dos arquivos) batem com os IDs do CSV experimental.")
        if len(df_full) == 0: return

    # 4. Análise de Correlação
    metricas = ['minimizedAffinity', 'CNNscore', 'CNNaffinity']

    print("\n--- CORRELAÇÕES COM pIC50 ---")

    for metrica in metricas:
        clean_df = df_full.dropna(subset=['pIC50', metrica])
        if len(clean_df) > 2:
            r_p, p_p = pearsonr(clean_df['pIC50'], clean_df[metrica])
            r_s, p_s = spearmanr(clean_df['pIC50'], clean_df[metrica])
            print(f"{metrica:.<20} Pearson R: {r_p:.4f} (p={p_p:.4f}) | Spearman R: {r_s:.4f}")
        else:
            print(f"{metrica}: Dados insuficientes.")

    # 5. Gráfico
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_full, x='pIC50', y='CNNaffinity', hue='CNNscore', palette='viridis', size='CNNscore', sizes=(20, 200))
        sns.regplot(data=df_full, x='pIC50', y='CNNaffinity', scatter=False, color='red', line_kws={'linestyle':'--'})
        plt.title(f"Docking (CNNaffinity) vs Experimental (pIC50)")
        plt.xlabel("pIC50 Experimental")
        plt.ylabel("Gnina CNN Affinity")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_PLOT)
        print(f"\nGráfico salvo em: {OUTPUT_PLOT}")
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")

    # 6. Top 15 Consenso
    df_full['Rank_Exp'] = df_full['pIC50'].rank(ascending=False)
    df_full['Rank_Dock'] = df_full['CNNaffinity'].rank(ascending=False)
    df_full['Rank_Consenso'] = df_full['Rank_Exp'] + df_full['Rank_Dock']

    df_final = df_full.sort_values('Rank_Consenso').head(15)

    print("\n--- TOP 15 CANDIDATOS (CONSENSO) ---")
    print(df_final[['ID', 'pIC50', 'CNNaffinity', 'Rank_Consenso']].to_string(index=False))

    df_full.sort_values('Rank_Consenso').to_csv(FINAL_TABLE, index=False)
    print(f"\nTabela completa salva em: {FINAL_TABLE}")

if __name__ == "__main__":
    main()
