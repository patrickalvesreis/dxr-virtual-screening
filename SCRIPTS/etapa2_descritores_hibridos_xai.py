import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- Hotfix para Mordred com NumPy 2.x ---
# Mordred tenta fazer: from numpy import product
# Em versões novas, 'product' pode não existir, mas 'np.prod' existe.
if not hasattr(np, "product"):
    np.product = np.prod

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold

from mordred import Calculator, descriptors


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Diretório base = UM NÍVEL ACIMA da pasta scripts/
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_SDF = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pdb_template" / "ligantes_corrigidos_v5.sdf"

OUTPUT_DIR = BASE_DIR / "03_Descritores_Hibridos"
OUTPUT_RAW = OUTPUT_DIR / "descritores_completo_bruto.csv"
OUTPUT_CLEAN = OUTPUT_DIR / "descritores_completo_limpo.csv"

# Config do fingerprint
MORGAN_NBITS = 1024  # 1024 bits é um bom compromisso (pode subir pra 2048 depois)
MORGAN_RADIUS = 2    # ECFP4 (raio=2)


# ============================================================
# FUNÇÕES DE DESCRITORES 3D (RDKit)
# ============================================================

def get_3d_descriptors(mol):
    """
    Calcula descritores 3D que exigem conformação.
    Se não tiver conformero 3D, retorna NaNs.
    """
    desc = {}

    if mol.GetNumConformers() == 0:
        # mesmíssimo conjunto de chaves usado quando dá certo
        return {
            "RadiusOfGyration": np.nan,
            "InertialShapeFactor": np.nan,
            "SpherocityIndex": np.nan,
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "PBF": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "PMI1": np.nan,
            "PMI2": np.nan,
            "PMI3": np.nan,
        }

    try:
        desc["RadiusOfGyration"]    = rdMolDescriptors.CalcRadiusOfGyration(mol)
        desc["InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
        desc["SpherocityIndex"]     = rdMolDescriptors.CalcSpherocityIndex(mol)
        desc["Asphericity"]         = rdMolDescriptors.CalcAsphericity(mol)
        desc["Eccentricity"]        = rdMolDescriptors.CalcEccentricity(mol)
        desc["PBF"]                 = rdMolDescriptors.CalcPBF(mol)  # Plane of Best Fit
        desc["NPR1"]                = rdMolDescriptors.CalcNPR1(mol)
        desc["NPR2"]                = rdMolDescriptors.CalcNPR2(mol)
        desc["PMI1"]                = rdMolDescriptors.CalcPMI1(mol)
        desc["PMI2"]                = rdMolDescriptors.CalcPMI2(mol)
        desc["PMI3"]                = rdMolDescriptors.CalcPMI3(mol)
    except Exception as e:
        # Se alguma geometria maluca quebrar a matemática, retorna NaNs
        desc = {
            "RadiusOfGyration": np.nan,
            "InertialShapeFactor": np.nan,
            "SpherocityIndex": np.nan,
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "PBF": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "PMI1": np.nan,
            "PMI2": np.nan,
            "PMI3": np.nan,
        }

    return desc


# ============================================================
# FUNÇÃO DE FINGERPRINT (Morgan / ECFP)
# ============================================================

def get_morgan_fingerprints(mol, n_bits=1024, radius=2):
    """
    Gera fingerprints de Morgan (ECFP-like).
    Essenciais para XAI visual (mapear bits em subestruturas).

    Cada bit vai virar uma coluna binária FP_0, FP_1, ..., FP_(n_bits-1).
    """
    fp_dict = {}

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    for i in range(n_bits):
        fp_dict[f"FP_{i}"] = fp.GetBit(i)

    return fp_dict


# ============================================================
# LIMPEZA (Variance + Correlação)
# ============================================================

def clean_dataset(df, corr_threshold=0.95):
    """
    Aplica filtros de:
      - Variância zero (remove colunas constantes)
      - Correlação alta (> corr_threshold)
    Mantém colunas de ID/SMILES como metadados.
    """
    print(f"\n[LIMPEZA] Iniciando limpeza de dados. Shape original: {df.shape}")

    cols_meta = [c for c in df.columns if c in ["ID", "SMILES", "LigandID", "Name", "Source"]]
    df_meta = df[cols_meta]
    df_num = df.drop(columns=cols_meta).select_dtypes(include=[np.number])

    # Preenche NaNs com 0 (principalmente para descritores Mordred/3D que falham)
    df_num = df_num.fillna(0)

    # 1. Remover colunas com variância zero
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_num)
    cols_var = df_num.columns[selector.get_support()]
    df_num = df_num[cols_var]
    print(f"[LIMPEZA] Colunas após remover variância zero: {df_num.shape[1]}")

    # 2. Remover colunas com alta correlação
    corr_matrix = df_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    df_num = df_num.drop(columns=to_drop)
    print(f"[LIMPEZA] Colunas após remover alta correlação (>{corr_threshold}): {df_num.shape[1]}")

    df_final = pd.concat([df_meta, df_num], axis=1)
    return df_final


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_SDF.exists():
        print(f"[ERRO] Arquivo não encontrado: {INPUT_SDF}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- INICIANDO CÁLCULO: MORDRED (2D) + RDKit (3D) + Morgan (FP) ---")

    # 1. Configurar calculadora Mordred (2D)
    calc_mordred = Calculator(descriptors, ignore_3D=True)
    print(f"[INFO] Descritores Mordred registrados (2D): {len(calc_mordred.descriptors)}")

    # 2. Ler SDF (mantendo H do docking / template)
    suppl = Chem.SDMolSupplier(str(INPUT_SDF), removeHs=False)

    data_rows = []

    print(f"[INFO] Lendo moléculas de: {INPUT_SDF.name}")
    for i, mol in enumerate(suppl):
        if mol is None:
            continue

        row = {}

        # ----- Identificadores -----
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        row["ID"] = mol_name

        try:
            row["SMILES"] = Chem.MolToSmiles(mol)
        except Exception:
            row["SMILES"] = ""

        # ----- A) MORDRED (2D) -----
        try:
            mordred_vals = calc_mordred(mol)
            row.update({
                str(k): (float(v) if not isinstance(v, (Exception, type(None))) else np.nan)
                for k, v in mordred_vals.items()
            })
        except Exception as e:
            print(f"[AVISO] Erro Mordred na molécula {mol_name}: {e}")

        # ----- B) RDKit 3D -----
        vals_3d = get_3d_descriptors(mol)
        row.update(vals_3d)

        # ----- C) Morgan Fingerprints (FP_0 ... FP_N) -----
        fp_dict = get_morgan_fingerprints(
            mol,
            n_bits=MORGAN_NBITS,
            radius=MORGAN_RADIUS,
        )
        row.update(fp_dict)

        data_rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"[INFO] Moléculas processadas: {i + 1}...")

    # 3. DataFrame completo
    df_full = pd.DataFrame(data_rows)

    print(f"[INFO] Shape bruto (antes da limpeza): {df_full.shape}")
    print(f"[INFO] Salvando CSV bruto em: {OUTPUT_RAW}")
    df_full.to_csv(OUTPUT_RAW, index=False)

    # 4. Limpeza estatística
    df_clean = clean_dataset(df_full, corr_threshold=0.95)

    print(f"[INFO] Salvando CSV limpo em: {OUTPUT_CLEAN}")
    df_clean.to_csv(OUTPUT_CLEAN, index=False)

    print("-" * 40)
    print("CONCLUÍDO")
    print(f"Bruto salvo em : {OUTPUT_RAW}")
    print(f"Limpo salvo em : {OUTPUT_CLEAN}")
    print(f"Redução colunas: {df_full.shape[1]} -> {df_clean.shape[1]}")


if __name__ == "__main__":
    main()
