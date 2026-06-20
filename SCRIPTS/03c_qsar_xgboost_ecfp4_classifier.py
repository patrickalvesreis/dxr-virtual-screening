# [DIR: ~/chem_gnn_pixi/PROJETO_DXR_ML_GNN/02_ML_Classico_Baselines/scripts]
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- CONFIGURAÇÕES ---
BASE_DIR = "../../"
INPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "ecfp4_features.csv")
THRESHOLD_PIC50 = 6.0  # Limiar Ativo/Inativo

def run_classifier():
    print("--- Classificação ECFP4 (XGBoost) ---")
    if not os.path.exists(INPUT_FILE):
        print("Arquivo ECFP4 não encontrado.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Features (bits) e Target
    features = [c for c in df.columns if c.startswith('bit_')]
    X = df[features]
    y = (df['pIC50'] >= THRESHOLD_PIC50).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Treino
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_res, y_train_res)

    # Predições
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n>>> RESULTADOS ECFP4 (CLASSIFICAÇÃO) <<<")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_classifier()
