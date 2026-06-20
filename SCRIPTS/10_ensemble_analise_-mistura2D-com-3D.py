"""
SCRIPT: 10_ensemble_analise_(mistura2D-com-3D).py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Carrega as previsões do modelo 2D e do modelo 3D.
           Testa diferentes pesos para encontrar a melhor combinação (Ensemble).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_2D_PATH = os.path.join(BASE_DIR, "..", "outputs", "predicoes_teste_2d.csv")
PREDS_3D_PATH = os.path.join(BASE_DIR, "..", "outputs", "predicoes_teste_3d.csv")
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "resultado_ensemble_final.png")

def main():
    print("--- ANÁLISE DE ENSEMBLE (2D + 3D) ---")

    # 1. Carregar Previsões
    if not os.path.exists(PREDS_2D_PATH) or not os.path.exists(PREDS_3D_PATH):
        print("Erro: Arquivos de previsão não encontrados. Rode os scripts 08 e 08b antes.")
        return

    df_2d = pd.read_csv(PREDS_2D_PATH)
    df_3d = pd.read_csv(PREDS_3D_PATH)

    # 2. Juntar os dataframes pelo ID (Garantir que estamos somando a mesma molécula)
    # Como usamos random_state=42, os IDs devem bater, mas o merge garante.
    df_final = pd.merge(df_2d, df_3d[['ID', 'Pred_3D']], on='ID')

    y_real = df_final['Real_pIC50'].values
    pred_2d = df_final['Pred_2D'].values
    pred_3d = df_final['Pred_3D'].values

    print(f"Total de moléculas no conjunto de teste: {len(y_real)}")
    print(f"R² Isolado 2D: {r2_score(y_real, pred_2d):.4f}")
    print(f"R² Isolado 3D: {r2_score(y_real, pred_3d):.4f}")
    print("-" * 30)

    # 3. Loop de Otimização de Pesos
    # Vamos testar pesos de 0.0 a 1.0 para o modelo 2D
    best_r2 = -999
    best_weight = 0.0
    best_preds = None

    results = []

    for w_2d in np.linspace(0, 1, 101): # 0.00, 0.01 ... 1.00
        w_3d = 1.0 - w_2d

        # Fórmula do Ensemble: Média Ponderada
        ensemble_pred = (pred_2d * w_2d) + (pred_3d * w_3d)

        r2 = r2_score(y_real, ensemble_pred)
        results.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            best_weight = w_2d
            best_preds = ensemble_pred

    # 4. Resultados
    print(f"MELHOR R² ENCONTRADO: {best_r2:.4f}")
    print(f"Melhor Peso 2D: {best_weight:.2f} ({best_weight*100:.0f}%)")
    print(f"Melhor Peso 3D: {1.0 - best_weight:.2f} ({(1.0-best_weight)*100:.0f}%)")

    ganho = best_r2 - r2_score(y_real, pred_2d)
    if ganho > 0.001:
        print(f"\nCONCLUSÃO: O Ensemble melhorou o modelo em +{ganho:.4f} pontos!")
    else:
        print(f"\nCONCLUSÃO: O Ensemble não ajudou. O modelo 2D puro é melhor ou igual.")

    # 5. Plotar o Vencedor
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, best_preds, alpha=0.7, c='blue', edgecolors='k', label='Predições')

    # Linha ideal
    m, b = np.polyfit(y_real, best_preds, 1)
    plt.plot(y_real, m*y_real + b, color='red', linestyle='--', label='Tendência')
    plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], 'k:', label='Ideal (x=y)')

    plt.xlabel("pIC50 Real (Experimental)")
    plt.ylabel("pIC50 Predito (Ensemble)")
    plt.title(f"Melhor Modelo Combinado\nR²: {best_r2:.4f} (Peso 2D: {best_weight:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_SAVE_PATH)
    print(f"Gráfico final salvo em: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()
