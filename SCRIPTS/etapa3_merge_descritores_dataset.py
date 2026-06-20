import sys
from pathlib import Path
import pandas as pd

# ================================
# CONFIG PATHS
# ================================
SCRIPT_DIR = Path(__file__).resolve().parent
DOCK_DIR = SCRIPT_DIR.parent                  # .../docking_boltz-2_V3
ROOT_DIR = DOCK_DIR.parent                    # .../00_Dados_Originais

# Arquivo de descritores (saída da etapa 2)
DESCR_DIR = DOCK_DIR / "03_Descritores_Hibridos"
DESCR_FILE = DESCR_DIR / "descritores_completo_limpo.csv"

# Dataset CLEAN original (BindingDB + Ki etc.)
DATASET_FILE = ROOT_DIR / "dataset_dxr_v2_CLEAN.csv"

# Saída final para ML
OUT_DIR = DOCK_DIR / "04_Datasets_ML"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FULL = OUT_DIR / "dxr_dataset_descritores_xai.csv"

# >>>>> AJUSTE AQUI O NOME DA COLUNA TARGET (y) NO SEU CSV CLEAN <<<<<
# exemplos comuns: "Ki_nM", "IC50_nM", "pKi", "pIC50"
Y_COLUMN = "Ki_nM"   # TROCAR se o nome for outro!


def main():
    # -------------------------------
    # 1) Carregar descritores
    # -------------------------------
    if not DESCR_FILE.exists():
        print(f"[ERRO] Arquivo de descritores não encontrado: {DESCR_FILE}")
        sys.exit(1)

    df_desc = pd.read_csv(DESCR_FILE)
    print(f"[INFO] Descritores carregados: {df_desc.shape}")

    if "ID" not in df_desc.columns:
        print("[ERRO] A coluna 'ID' não existe em descritores_completo_limpo.csv.")
        sys.exit(1)

    # -------------------------------
    # 2) Carregar dataset CLEAN
    # -------------------------------
    if not DATASET_FILE.exists():
        print(f"[ERRO] Dataset CLEAN não encontrado: {DATASET_FILE}")
        sys.exit(1)

    df_clean = pd.read_csv(DATASET_FILE)
    print(f"[INFO] Dataset CLEAN carregado: {df_clean.shape}")
    print("[INFO] Colunas dataset CLEAN:", df_clean.columns.tolist())

    # Conferir coluna de ID
    if "BindingDB MonomerID" not in df_clean.columns:
        print("[ERRO] Coluna 'BindingDB MonomerID' não encontrada em dataset_dxr_v2_CLEAN.csv.")
        sys.exit(1)

    # Conferir coluna alvo (y)
    if Y_COLUMN not in df_clean.columns:
        print(f"[ERRO] Coluna alvo '{Y_COLUMN}' não encontrada no dataset CLEAN. Ajuste Y_COLUMN no script.")
        sys.exit(1)

    # -------------------------------
    # 3) Preparar dataframes para merge
    # -------------------------------
    df_clean_id_y = df_clean[["BindingDB MonomerID", Y_COLUMN]].copy()
    df_clean_id_y = df_clean_id_y.rename(columns={"BindingDB MonomerID": "ID"})

    print(f"[INFO] Subconjunto CLEAN (ID + y): {df_clean_id_y.shape}")

    # -------------------------------
    # 4) Merge (inner join) por ID
    # -------------------------------
    df_merged = pd.merge(df_clean_id_y, df_desc, on="ID", how="inner")

    print("-" * 40)
    print(f"[MERGE] df_clean_id_y: {df_clean_id_y.shape}")
    print(f"[MERGE] df_desc      : {df_desc.shape}")
    print(f"[MERGE] df_merged    : {df_merged.shape}")

    # Conferir se algum ID ficou de fora
    missing_in_desc = set(df_clean_id_y["ID"]) - set(df_desc["ID"])
    missing_in_clean = set(df_desc["ID"]) - set(df_clean_id_y["ID"])

    print(f"[CHECAGEM] IDs no CLEAN e não nos descritores: {len(missing_in_desc)}")
    if len(missing_in_desc) > 0:
        print("  exemplos:", list(missing_in_desc)[:10])

    print(f"[CHECAGEM] IDs nos descritores e não no CLEAN: {len(missing_in_clean)}")
    if len(missing_in_clean) > 0:
        print("  exemplos:", list(missing_in_clean)[:10])

    # -------------------------------
    # 5) Salvar dataset final
    # -------------------------------
    df_merged.to_csv(OUT_FULL, index=False)
    print("-" * 40)
    print(f"[OK] Dataset final para ML/XAI salvo em: {OUT_FULL}")
    print(f"[INFO] Colunas totais (incluindo ID + y): {df_merged.shape[1]}")
    print(f"[INFO] Linhas (ligantes com y + descritores): {df_merged.shape[0]}")


if __name__ == "__main__":
    main()

