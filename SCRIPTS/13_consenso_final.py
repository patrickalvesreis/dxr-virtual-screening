"""
SCRIPT: 13_consenso_final.py
DESCRIÇÃO: Combina GNN e Docking (Average Ranking) para melhor performance.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_GNN = os.path.join(BASE_DIR, "..", "outputs", "predicoes_gnn.csv")
FILE_DOCK = os.path.join(BASE_DIR, "..", "outputs", "resultado_FINAL_docking.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "..", "outputs", "modelo_consenso.png")

def main():
    print("--- MODELO DE CONSENSO (GNN + DOCKING) ---")

    # 1. Carregar
    df_gnn = pd.read_csv(FILE_GNN)
    df_dock = pd.read_csv(FILE_DOCK)

    df_gnn['ID'] = df_gnn['ID'].astype(str)
    df_dock['ID'] = df_dock['ID'].astype(str)

    # 2. Merge
    # Precisamos apenas da colina CNN_Affinity do docking
    df_merge = pd.merge(df_gnn, df_dock[['ID', 'CNN_Affinity']], on='ID', how='inner')

    print(f"Moléculas no consenso: {len(df_merge)}")

    # 3. Normalizar Escalas (0 a 1) para poder somar
    scaler = MinMaxScaler()

    # GNN Pred (Quanto maior melhor)
    df_merge['Norm_GNN'] = scaler.fit_transform(df_merge[['pIC50_GNN']])

    # Docking CNN Affinity (Quanto maior melhor - pKd)
    df_merge['Norm_Dock'] = scaler.fit_transform(df_merge[['CNN_Affinity']])

    # 4. Consenso (Média Simples)
    # Você pode dar pesos: ex: 0.7*GNN + 0.3*Docking
    df_merge['Score_Final'] = (df_merge['Norm_GNN'] * 0.6) + (df_merge['Norm_Dock'] * 0.4)

    # 5. Avaliar Correlação com Real
    r2 = r2_score(df_merge['pIC50_Real'], df_merge['Score_Final']) # R² não faz sentido aqui pois Score é 0-1
    pearson = df_merge['pIC50_Real'].corr(df_merge['Score_Final'])

    print("-" * 30)
    print(f"CORRELAÇÃO FINAL (PEARSON): {pearson:.4f}")
    print("-" * 30)

    # Gráfico
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_merge['pIC50_Real'], y=df_merge['Score_Final'], hue=df_merge['Score_Final'], palette='viridis')
    plt.xlabel("pIC50 Experimental")
    plt.ylabel("Score Consenso (GNN + Docking)")
    plt.title(f"Consenso Final: Pearson = {pearson:.2f}")
    plt.savefig(OUTPUT_PLOT)
    print(f"Gráfico salvo em: {OUTPUT_PLOT}")

# ... (código anterior igual) ...

    # 5. Avaliar Correlação e R²
    pearson = df_merge['pIC50_Real'].corr(df_merge['Score_Final'])

    # Cálculo do R² da regressão linear entre Real e Consenso
    # (Importante: R² score direto do sklearn pode ser negativo se a escala não for linear perfeita,
    #  então o quadrado de Pearson é a métrica mais honesta para "força da relação linear")
    r2_linear = pearson ** 2

    print("-" * 30)
    print(f"CORRELAÇÃO DE PEARSON (r): {pearson:.4f}")
    print(f"COEFICIENTE DE DETERMINAÇÃO (R²): {r2_linear:.4f}")
    print("-" * 30)

    # Gráfico com o valor de R² no título
    plt.figure(figsize=(8,6))
    sns.regplot(x=df_merge['pIC50_Real'], y=df_merge['Score_Final'],
                scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}, color='purple')
    plt.xlabel("pIC50 Experimental")
    plt.ylabel("Score Consenso Normalizado")
    plt.title(f"Consenso Final: Pearson (r) = {pearson:.2f} | R² = {r2_linear:.2f}")
    plt.savefig(OUTPUT_PLOT)
    print(f"Gráfico salvo em: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
