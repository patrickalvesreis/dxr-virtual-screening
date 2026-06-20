"""
SCRIPT: 12_ranking_final_unificado.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO:
  1. Limpa duplicatas do experimental.
  2. Junta: Experimental + Predição ML (2D) + Docking (CNNscore).
  3. Gera um RANKING FINAL baseado no Consenso.
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"

# Arquivos de Entrada
EXP_CSV = os.path.join(PROJECT_DIR, "outputs", "descritores_basicos.csv")
DOCK_CSV = os.path.join(PROJECT_DIR, "outputs", "resultados_docking_gnina.csv")
# Atenção: Precisamos das predições do ML para TODOS, não só o teste.
# Mas como geramos predicoes_teste_2d.csv só pro teste, vamos usar o docking como filtro principal
# e cruzar com o experimental limpo.

OUTPUT_CSV = os.path.join(PROJECT_DIR, "outputs", "TABELA_MESTRA_FINAL.csv")

def main():
    print("--- GERANDO TABELA MESTRA DE CONSENSO ---")

    # 1. Carregar Experimental e Limpar Duplicatas
    if not os.path.exists(EXP_CSV): return
    df_exp = pd.read_csv(EXP_CSV)

    # IMPORTANTE: Agrupar por ID e tirar a média do pIC50
    print(f"Experimental Original: {len(df_exp)} linhas")
    df_exp['ID'] = df_exp['ID'].astype(str)
    df_exp = df_exp.groupby('ID').agg({
        'pIC50': 'mean',
        'SMILES': 'first' # Pega o primeiro SMILES encontrado
    }).reset_index()
    print(f"Experimental Limpo (Sem duplicatas): {len(df_exp)} moléculas únicas")

    # 2. Carregar Docking
    if not os.path.exists(DOCK_CSV): return
    df_dock = pd.read_csv(DOCK_CSV)
    df_dock['ID'] = df_dock['ID'].astype(str)

    # 3. Merge (Juntar)
    df_final = pd.merge(df_exp, df_dock[['ID', 'CNNscore', 'CNNaffinity', 'minimizedAffinity']], on='ID', how='inner')
    print(f"Moléculas com Dados Completos: {len(df_final)}")

    # 4. Criar Sistema de Pontuação (Scoring)
    # Vamos usar Ranks Percentuais (0 a 1, onde 1 é o melhor)

    # pIC50: Quanto maior, melhor
    df_final['Score_Exp'] = df_final['pIC50'].rank(pct=True)

    # CNNscore: Quanto maior, melhor (Correlação positiva R=0.29)
    df_final['Score_Dock'] = df_final['CNNscore'].rank(pct=True)

    # CNNaffinity: Quanto maior, melhor
    df_final['Score_Aff'] = df_final['CNNaffinity'].rank(pct=True)

    # 5. Score Consenso (Média dos Scores)
    # Vamos dar peso igual para Experimental e Docking (CNNscore que foi o melhor)
    # Score Final = (Rank do pIC50 + Rank do Docking) / 2
    df_final['Consenso_Score'] = (df_final['Score_Exp'] + df_final['Score_Dock']) / 2

    # 6. Ordenar e Salvar
    df_final = df_final.sort_values('Consenso_Score', ascending=False)

    # Selecionar colunas bonitas
    cols = ['ID', 'SMILES', 'pIC50', 'CNNscore', 'CNNaffinity', 'Consenso_Score']
    df_export = df_final[cols].round(4)

    print("\n--- TOP 10 MELHORES CANDIDATOS (Consenso Docking + Exp) ---")
    print(df_export.head(10).to_string(index=False))

    df_export.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTabela Mestra salva em: {OUTPUT_CSV}")

    # Análise Rápida do Top 10%
    top_10_percent = df_final.head(int(len(df_final)*0.1))
    avg_pic50_top = top_10_percent['pIC50'].mean()
    avg_pic50_all = df_final['pIC50'].mean()

    print(f"\nCuriosidade:")
    print(f"Média de pIC50 geral: {avg_pic50_all:.2f}")
    print(f"Média de pIC50 no Top 10% do Docking: {avg_pic50_top:.2f}")
    if avg_pic50_top > avg_pic50_all:
        print("-> O Docking conseguiu enriquecer a seleção com moléculas mais potentes!")

if __name__ == "__main__":
    main()
