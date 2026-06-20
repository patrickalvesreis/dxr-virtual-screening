"""
SCRIPT: 09_salvacao_xgboost.py
DESCRIÇÃO: Tenta salvar o dataset 2D+3D usando XGBoost e Seleção de Features agressiva.
REQUER: pip install xgboost (se não tiver no pixi, o script tenta usar sklearn puro se falhar)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Tenta importar XGBoost, senão usa GradientBoosting do sklearn
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Aviso: XGBoost não instalado. Usando GradientBoosting do Scikit-Learn.")

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "outputs", "super_dataset.npz")
PLOT_PATH = os.path.join(BASE_DIR, "..", "outputs", "resultado_xgboost.png")

def main():
    print("--- OPERAÇÃO RESGATE: XGBOOST & FEATURE SELECTION ---")

    # 1. Carregar o Dataset Gigante
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Rode o script 07 primeiro!")

    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']

    # Limpeza básica de NaNs/Infinitos que podem vir do 3D
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Dimensão Original: {X.shape} (Muita coluna, pouca linha!)")

    # 2. Passo Crítico: Limpeza de Features Inúteis
    print("1. Removendo features com variância zero (colunas constantes)...")
    selector = VarianceThreshold(threshold=0.01) # Remove colunas que quase não mudam
    X_clean = selector.fit_transform(X)
    print(f"   -> Reduzido para: {X_clean.shape}")

    # 3. Split (Mesma seed para comparar honestamente)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42
    )

    # 4. Configurar Modelo Robusto
    if HAS_XGB:
        print("2. Treinando XGBoost Regressor (Otimizado para dados tabulares)...")
        model = xgb.XGBRegressor(
            n_estimators=1000,     # Muitas árvores
            learning_rate=0.01,    # Aprendizado lento e cuidadoso
            max_depth=6,           # Profundidade média
            subsample=0.7,         # Usa apenas 70% dos dados por árvore (evita overfitting)
            colsample_bytree=0.5,  # Usa apenas 50% das features por árvore (força variedade)
            n_jobs=-1,
            random_state=42
        )
    else:
        print("2. Treinando Gradient Boosting (Sklearn)...")
        model = GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )

    # Treino
    model.fit(X_train, y_train)

    # 5. Avaliação
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    print("-" * 30)
    print(f"RESULTADO FINAL (XGBoost):")
    print(f"Train R²: {r2_train:.4f} (Deve ser alto)")
    print(f"Test R²:  {r2_test:.4f}")
    print(f"Test RMSE:{rmse_test:.4f}")
    print("-" * 30)

    if r2_test > 0.49:
        print("✅ SUCESSO: Batemos a GNN!")
    else:
        print("❌ CONCLUSÃO: O dataset é pequeno demais para ML puro.")
        print("   -> A GNN (R² 0.49) continua sendo a melhor abordagem.")

    # 6. Gráfico
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Scatter plot
    sns.scatterplot(x=y_test, y=y_pred, color='darkgreen', s=100, alpha=0.7, label='Teste')

    # Linha Ideal
    min_val = min(y_test.min(), y_pred.min()) - 0.5
    max_val = max(y_test.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

    plt.xlabel("pIC50 Real")
    plt.ylabel("pIC50 Predito (XGBoost)")
    plt.title(f"XGBoost (Features 2D+3D) | R² = {r2_test:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Gráfico salvo: {PLOT_PATH}")

if __name__ == "__main__":
    main()
