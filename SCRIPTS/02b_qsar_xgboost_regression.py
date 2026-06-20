# [DIR: ~/chem_gnn_pixi/PROJETO_DXR_ML_GNN/02_ML_Classico_Baselines/scripts]
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# --- CONFIGURAÇÕES ---
BASE_DIR = "../../"
INPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "descritores_rdkit_completo.csv")
OUTPUT_MODELS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "models")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "results")
OUTPUT_PLOTS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "plots")

# Criar pastas
for d in [OUTPUT_MODELS, OUTPUT_RESULTS, OUTPUT_PLOTS]:
    os.makedirs(d, exist_ok=True)

def run_regression():
    print("--- Carregando dados para Regressão ---")
    df = pd.read_csv(INPUT_FILE)

    # Remover colunas que não são features
    cols_to_drop = ['ID', 'SMILES', 'IC50_nM', 'pIC50', 'SMILES_Limpo', 'IC50_nM_Original']
    features = [c for c in df.columns if c not in cols_to_drop and pd.api.types.is_numeric_dtype(df[c])]

    df_model = df.dropna(subset=features + ['pIC50']) # Garante que temos pIC50

    X = df_model[features]
    y = df_model['pIC50'] # Agora o alvo é o número contínuo!

    print(f"Total de amostras: {len(X)}")

    # Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # OBS: Não usamos SMOTE em regressão padrão

    print("--- Treinando XGBoost Regressor ---")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Avaliação
    print("--- Avaliando no Teste ---")
    y_pred = model.predict(X_test)

    # Métricas de Regressão
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n>>> RESULTADOS REGRESSÃO <<<")
    print(f"R² (R-Quadrado): {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("-" * 30)

    # Salvar Resultados
    metrics = {"r2": r2, "rmse": rmse, "mae": mae}
    with open(os.path.join(OUTPUT_RESULTS, "metrics_regression.json"), "w") as f:
        json.dump(metrics, f)

    joblib.dump(model, os.path.join(OUTPUT_MODELS, "xgboost_regressor.pkl"))

    # Gráfico Predito vs Experimental (O favorito dos professores)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k')

    # Linha ideal (x=y)
    min_val = min(y_test.min(), y_pred.min()) - 0.5
    max_val = max(y_test.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

    plt.xlabel('Experimental pIC50')
    plt.ylabel('Predito pIC50 (XGBoost)')
    plt.title(f'Regressão XGBoost\n$R^2$ = {r2:.3f} | RMSE = {rmse:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "regressao_predito_vs_experimental.png"))
    plt.close()

    print(f"Gráfico de correlação salvo em: {os.path.join(OUTPUT_PLOTS, 'regressao_predito_vs_experimental.png')}")

if __name__ == "__main__":
    run_regression()
