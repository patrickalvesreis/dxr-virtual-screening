"""
SCRIPT: 11_analise_correlacao_docking.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Cruza os resultados do Docking (Gnina) com o pIC50 experimental.
           Verifica se o docking consegue prever a atividade biológica.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Caminho dos resultados do Gnina (que geramos no script 09 final)
DOCKING_CSV = os.path.join(BASE_DIR, "..", "outputs", "resultados_docking_gnina.csv")
# Caminho dos dados experimentais
EXP_CSV = os.path.join(BASE_DIR, "..", "outputs", "descritores_basicos.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "..", "outputs", "correlacao_docking_vs_pIC50.png")
FINAL_TABLE = os.path.join(BASE_DIR, "..", "outputs", "tabela_final_ranking_unificado.csv")

def main():
    print("--- ANÁLISE FINAL: DOCKING vs EXPERIMENTAL ---")

    # 1. Carregar Dados
    if not os.path.exists(DOCKING_CSV):
        print(f"ERRO: Arquivo {DOCKING_CSV} não encontrado.")
        return

    df_dock = pd.read_csv(DOCKING_CSV) # Colunas: ID, minimizedAffinity, CNNscore, CNNaffinity
    df_exp = pd.read_csv(EXP_CSV)      # Colunas: ID, pIC50, SMILES...

    # Converter ID para string para garantir o merge correto
    df_dock['ID'] = df_dock['ID'].astype(str)
    df_exp['ID'] = df_exp['ID'].astype(str)

    # 2. Merge (Juntar tabelas)
    df_full = pd.merge(df_exp[['ID', 'pIC50', 'SMILES']],
                       df_dock[['ID', 'minimizedAffinity', 'CNNscore', 'CNNaffinity']],
                       on='ID', how='inner')

    print(f"Total de moléculas com Docking + pIC50: {len(df_full)}")

    if len(df_full) < 10:
        print("Poucos dados para correlação. Verifique se os IDs batem.")
        return

    # 3. Análise de Correlação
    # CNNaffinity costuma ser o melhor preditor do Gnina
    metricas = ['minimizedAffinity', 'CNNscore', 'CNNaffinity']

    print("\n--- CORRELAÇÕES COM pIC50 ---")
    best_metric = 'CNNaffinity'
    best_r = -1

    for metrica in metricas:
        # Remover NaNs
        clean_df = df_full.dropna(subset=['pIC50', metrica])

        # Pearson (Linear)
        r_p, p_p = pearsonr(clean_df['pIC50'], clean_df[metrica])
        # Spearman (Rank - Ordem)
        r_s, p_s = spearmanr(clean_df['pIC50'], clean_df[metrica])

        print(f"{metrica:.<20} Pearson R: {r_p:.4f} (p={p_p:.4f}) | Spearman R: {r_s:.4f}")

        if abs(r_p) > best_r:
            best_r = abs(r_p)
            best_metric = metrica

    # 4. Gráfico de Dispersão (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_full, x='pIC50', y='CNNaffinity', hue='CNNscore', palette='viridis', size='CNNscore', sizes=(20, 200))

    # Linha de tendência
    sns.regplot(data=df_full, x='pIC50', y='CNNaffinity', scatter=False, color='red', line_kws={'linestyle':'--'})

    plt.title(f"Correlação Docking vs Experimental\n(Pearson R: {df_full['pIC50'].corr(df_full['CNNaffinity']):.3f})")
    plt.xlabel("pIC50 Experimental (Atividade Real)")
    plt.ylabel("Gnina CNN Affinity (Predição Estrutural)")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nGráfico salvo em: {OUTPUT_PLOT}")

    # 5. Top 10 Moléculas "Consenso" (Boas no Docking E no Experimental)
    # Vamos normalizar os scores para somar
    # pIC50: quanto maior melhor
    # CNNaffinity: quanto maior melhor
    # minimizedAffinity: quanto MENOR (mais negativo) melhor -> Inverteremos

    df_full['Rank_Exp'] = df_full['pIC50'].rank(ascending=False)
    df_full['Rank_Dock'] = df_full['CNNaffinity'].rank(ascending=False)

    # O Rank Consenso é a soma dos ranks (quanto menor a soma, melhor a posição média)
    df_full['Rank_Consenso'] = df_full['Rank_Exp'] + df_full['Rank_Dock']

    df_final = df_full.sort_values('Rank_Consenso').head(15)

    print("\n--- TOP 15 CANDIDATOS (CONSENSO EXP + DOCKING) ---")
    print(df_final[['ID', 'pIC50', 'CNNaffinity', 'Rank_Consenso']].to_string(index=False))

    df_full.sort_values('Rank_Consenso').to_csv(FINAL_TABLE, index=False)
    print(f"\nTabela completa salva em: {FINAL_TABLE}")

if __name__ == "__main__":
    main()
