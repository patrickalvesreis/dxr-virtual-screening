"""
SCRIPT: 05_comparar_rf_scikit.py
DESCRIÇÃO: Treina Random Forest (ECFP4) e compara com GNN, usando gráficos Seaborn.
ENTRADA: outputs/ecfp_deepchem.csv
SAÍDA: outputs/comparacao_rf_seaborn.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "outputs", "ecfp_deepchem.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "..", "outputs", "comparacao_rf_seaborn.png")

def main():
    print("--- RANDOM FOREST vs GNN (BENCHMARK) ---")

    # 1. Carregar Dados
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"Arquivo não encontrado: {INPUT_FILE}")

    print("Carregando dados ECFP...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['pIC50'])

    # Features (Bits) e Target
    feat_cols = [c for c in df.columns if c.startswith('Bit_')]
    X = df[feat_cols].values
    y = df['pIC50'].values

    print(f"Dataset shape: {X.shape}")

    # 2. Split (random_state=42 garante a mesma divisão da GNN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Treinar Random Forest
    print("Treinando Random Forest (pode levar alguns segundos)...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,     # Controla overfitting
        n_jobs=-1,        # Usa todos os núcleos da CPU
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 4. Predições
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    # Métricas
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print("-" * 30)
    print(f"RESULTADOS RANDOM FOREST (ECFP4):")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²:  {r2_test:.4f}")
    print(f"Test MSE: {mse_test:.4f}")
    print("-" * 30)

    # 5. Plotar com Seaborn
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Gráfico de Regressão (Scatter + Linha de ajuste + Intervalo de Confiança)
    sns.regplot(
        x=y_test,
        y=y_pred,
        color="b",
        scatter_kws={'alpha':0.6, 's':60},
        line_kws={'color':'red', 'label':'Ajuste Linear'}
    )

    # Linha Ideal (x=y)
    min_val = min(y_test.min(), y_pred.min()) - 0.5
    max_val = max(y_test.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', lw=2, label='Ideal (Perfeito)')

    plt.xlabel("pIC50 Experimental (Real)", fontsize=12)
    plt.ylabel("pIC50 Predito (Random Forest)", fontsize=12)
    plt.title(f"Benchmark: Random Forest (R² = {r2_test:.2f}) vs Real", fontsize=14)
    plt.legend()
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Gráfico salvo em alta resolução: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
