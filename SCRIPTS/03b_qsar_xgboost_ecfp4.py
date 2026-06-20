# [DIR: ~/chem_gnn_pixi/PROJETO_DXR_ML_GNN/02_ML_Classico_Baselines/scripts]
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# --- CONFIGURAÇÕES ---
BASE_DIR = "../../"
# Agora lendo o arquivo de Fingerprints!
INPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "ecfp4_features.csv")

def run_test():
    print("--- Teste Rápido: XGBoost com ECFP4 ---")
    if not os.path.exists(INPUT_FILE):
        print("Gere os fingerprints primeiro (script 03)!")
        return

    df = pd.read_csv(INPUT_FILE)

    # Features são todas as colunas que começam com 'bit_'
    features = [c for c in df.columns if c.startswith('bit_')]
    X = df[features]
    y = df['pIC50']

    print(f"Features (bits): {len(features)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treino
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n>>> RESULTADO ECFP4 <<<")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_test()
