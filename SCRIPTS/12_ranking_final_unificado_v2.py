"""
SCRIPT: 12_ranking_final_unificado_v2.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Gera Tabela Mestra, mas com busca inteligente de arquivos perdidos nas pastas.
"""

import pandas as pd
import numpy as np
import os

# --- CONFIGURAÇÕES ---
# Caminho da pasta onde o script está rodando agora
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho Raiz do Projeto (Absoluto)
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"

# Onde salvar o resultado final (Na raiz para ser fácil de achar)
OUTPUT_CSV = os.path.join(PROJECT_DIR, "outputs", "TABELA_MESTRA_FINAL.csv")

def find_file(filename, search_paths):
    """Procura um arquivo em uma lista de pastas e retorna o primeiro encontrado."""
    print(f"  > Procurando '{filename}'...")
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            print(f"    [OK] Encontrado em: {full_path}")
            return full_path
        else:
            print(f"    [X] Não está em: {full_path}")
    return None

def main():
    print("--- GERANDO TABELA MESTRA (MODO DIAGNÓSTICO) ---")

    # Pastas onde vamos procurar os arquivos (Raiz/outputs e Script/../outputs)
    paths_to_search = [
        os.path.join(PROJECT_DIR, "outputs"),
        os.path.join(SCRIPT_DIR, "..", "outputs"),
        os.path.join(PROJECT_DIR, "Banco-de-dados-bindingDB", "Descritores", "outputs")
    ]

    # 1. Procurar Experimental
    print("\n1. Buscando dados experimentais (descritores_basicos.csv):")
    exp_file = find_file("descritores_basicos.csv", paths_to_search)

    if not exp_file:
        print("\nERRO CRÍTICO: Não achei 'descritores_basicos.csv' em lugar nenhum.")
        return

    # 2. Procurar Docking
    print("\n2. Buscando dados de Docking (resultados_docking_gnina.csv):")
    dock_file = find_file("resultados_docking_gnina.csv", paths_to_search)

    if not dock_file:
        print("\nERRO CRÍTICO: Não achei 'resultados_docking_gnina.csv' em lugar nenhum.")
        return

    # --- PROCESSAMENTO ---
    print("\n3. Processando dados...")

    # Carregar
    df_exp = pd.read_csv(exp_file)
    df_dock = pd.read_csv(dock_file)

    # Limpar Experimental (Média de duplicatas)
    print(f"   - Experimental Bruto: {len(df_exp)} linhas")
    df_exp['ID'] = df_exp['ID'].astype(str)
    df_exp = df_exp.groupby('ID').agg({
        'pIC50': 'mean',
        'SMILES': 'first'
    }).reset_index()
    print(f"   - Experimental Único: {len(df_exp)} moléculas")

    # Preparar Docking
    df_dock['ID'] = df_dock['ID'].astype(str)

    # Merge
    df_final = pd.merge(df_exp, df_dock[['ID', 'CNNscore', 'CNNaffinity']], on='ID', how='inner')
    print(f"   - Interseção (Exp + Docking): {len(df_final)} moléculas")

    if len(df_final) == 0:
        print("ERRO: O cruzamento deu zero. Verifique se os IDs são compatíveis.")
        return

    # Calcular Scores (Consenso)
    # Normaliza de 0 a 1 (Rank Percentual)
    df_final['Score_Exp'] = df_final['pIC50'].rank(pct=True)
    df_final['Score_Dock'] = df_final['CNNscore'].rank(pct=True)

    # Score Final (Média)
    df_final['Consenso_Score'] = (df_final['Score_Exp'] + df_final['Score_Dock']) / 2

    # Ordenar
    df_final = df_final.sort_values('Consenso_Score', ascending=False)

    # Seleção final
    cols = ['ID', 'pIC50', 'CNNscore', 'CNNaffinity', 'Consenso_Score']
    df_export = df_final[cols].round(4)

    print("\n--- TOP 10 MELHORES CANDIDATOS (Consenso) ---")
    print(df_export.head(10).to_string(index=False))

    # Estatística de Enriquecimento
    top_10_percent = df_final.head(int(len(df_final)*0.1))
    media_geral = df_final['pIC50'].mean()
    media_top = top_10_percent['pIC50'].mean()

    print(f"\n--- ANÁLISE DE ENRIQUECIMENTO ---")
    print(f"Média pIC50 Geral: {media_geral:.3f}")
    print(f"Média pIC50 no Top 10%: {media_top:.3f}")

    if media_top > media_geral:
        delta = ((media_top - media_geral) / media_geral) * 100
        print(f"RESULTADO: O Consenso melhorou a seleção em +{delta:.1f}%!")
    else:
        print("RESULTADO: O Consenso não enriqueceu a seleção.")

    df_export.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTabela salva em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
