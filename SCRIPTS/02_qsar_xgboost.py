import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# --- CONFIGURAÇÕES ---
# O script assume que está rodando de dentro de: 02_ML_Classico_Baselines/scripts
BASE_DIR = "../../"
INPUT_FILE = os.path.join(BASE_DIR, "01_Feature_Engineering", "data", "descritores_rdkit_completo.csv")
OUTPUT_MODELS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "models")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "results")
OUTPUT_PLOTS = os.path.join(BASE_DIR, "02_ML_Classico_Baselines", "plots")

# Definição de Ativo: pIC50 >= 6.0 (Equivalente a < 1000 nM ou 1 uM)
THRESHOLD_PIC50 = 6.0

# Criar pastas de saída se não existirem
for d in [OUTPUT_MODELS, OUTPUT_RESULTS, OUTPUT_PLOTS]:
    os.makedirs(d, exist_ok=True)

def run_pipeline():
    print("--- Carregando dados ---")
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO CRÍTICO: Arquivo não encontrado em: {INPUT_FILE}")
        print("Verifique se você moveu o arquivo 'descritores_rdkit_completo.csv' para '01_Feature_Engineering/data/'")
        return

    df = pd.read_csv(INPUT_FILE)

    # Colunas que NÃO são features para o modelo
    cols_to_drop = ['ID', 'SMILES', 'IC50_nM', 'pIC50', 'SMILES_Limpo', 'IC50_nM_Original']

    # Selecionar apenas colunas numéricas que sobraram (as features do RDKit)
    # Garante que remove colunas de texto extras se houverem
    features = [c for c in df.columns if c not in cols_to_drop and pd.api.types.is_numeric_dtype(df[c])]

    print(f"Features selecionadas: {len(features)}")

    # Limpar linhas com NaN nas features (o RDKit às vezes falha em 1 ou 2 descritores complexos)
    df_model = df.dropna(subset=features)

    X = df_model[features]
    # Criar target binário: 1 se Ativo, 0 se Inativo
    y = (df_model['pIC50'] >= THRESHOLD_PIC50).astype(int)

    print(f"Total de moléculas válidas: {len(X)}")
    print(f"Contagem de classes (Original): {y.value_counts().to_dict()}")

    if len(y.unique()) < 2:
        print("ERRO: O dataset só tem uma classe (só ativos ou só inativos). Não é possível treinar.")
        return

    # Split Treino (80%) / Teste (20%) - Estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Aplicar SMOTE apenas no TREINO para não vazar dados sintéticos no teste
    print("--- Aplicando SMOTE (Balanceamento) no treino ---")
    try:
        # k_neighbors=1 é seguro para datasets muito pequenos, aumente se tivermos >50 ativos
        k_neighbors = 1 if y_train.value_counts().min() < 6 else 5
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Classes pós-SMOTE: {y_train_res.value_counts().to_dict()}")
    except Exception as e:
        print(f"Aviso: SMOTE falhou ({e}). Treinando sem balanceamento (pode afetar performance na classe minoritária).")
        X_train_res, y_train_res = X_train, y_train

    # Treinar XGBoost
    print("--- Treinando XGBoost ---")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Avaliação no Teste (Dados Reais)
    print("--- Avaliando no Teste ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5 # Caso só tenha 1 classe no teste por azar

    f1 = f1_score(y_test, y_pred)

    print(f"\n>>> RESULTADOS FINAIS <<<")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score (Classe Ativa): {f1:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

    # Salvar artefatos
    print("Salvando resultados...")
    metrics = {"roc_auc": auc, "f1_score": f1}
    with open(os.path.join(OUTPUT_RESULTS, "metrics_xgboost.json"), "w") as f:
        json.dump(metrics, f)

    joblib.dump(model, os.path.join(OUTPUT_MODELS, "xgboost_model.pkl"))

    # Gerar Gráficos
    # 1. Matriz de Confusão
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão (Teste)")
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig(os.path.join(OUTPUT_PLOTS, "confusion_matrix.png"))
    plt.close()

    # 2. Curva ROC
    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("Curva ROC - XGBoost Baseline")
    plt.savefig(os.path.join(OUTPUT_PLOTS, "roc_curve.png"))
    plt.close()

    # 3. Importância das Features (Top 15)
    plt.figure(figsize=(10,6))
    sorted_idx = model.feature_importances_.argsort()[-15:]
    plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx])
    plt.title("Top 15 Descritores RDKit Importantes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "feature_importance.png"))
    plt.close()

    print(f"Pronto! Verifique os gráficos em: {OUTPUT_PLOTS}")

if __name__ == "__main__":
    run_pipeline()
