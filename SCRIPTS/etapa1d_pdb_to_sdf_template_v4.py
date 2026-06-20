from pathlib import Path
import csv
import sys
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# ONDE RODAR ESTE SCRIPT:
#   cd ~/1_PROJETO_DOCAGEM/Projeto_DXR/Banco-de-dados-bindingDB/Descritores/PROJETO_DXR_ML_GNN/00_Dados_Originais/docking_boltz-2_V3
#   (chem_gnn) python scripts/etapa1d_pdb_to_sdf_template_v4.py

BASE_DIR = Path.cwd()

PDB_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pdb_template"
LOG_CSV = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1d_pdb_to_sdf_template_v4.csv"

# Dataset CLEAN (fonte de verdade dos SMILES)
DATASET_CSV = BASE_DIR.parent / "dataset_dxr_v2_CLEAN.csv"

OUT_SDF = OUT_DIR / "ligantes_corrigidos_v4.sdf"

# Garante diretórios
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def validar_diretorio_execucao():
    """Verifica se estamos na raiz da pasta de docking."""
    if not (BASE_DIR / "01_Mapeamentos").exists():
        print("[ERRO CRÍTICO] Parece que você não está na raiz do projeto de docking.")
        print(f"Diretório atual: {BASE_DIR}")
        print("Certifique-se de rodar o script a partir da pasta 'docking_boltz-2_V3'.")
        sys.exit(1)


def detectar_coluna_id(fieldnames: list[str]) -> str | None:
    """
    Tenta detectar a coluna de ID no dataset CLEAN.
    Prioridade (case-insensitive):
      - contém 'monomer'
      - contém 'reactant_set'
      - 'ligand_id'
      - 'id' (fallback)
    """
    prioridades = ["monomer", "reactant_set", "ligand_id"]

    for padrao in prioridades:
        for col in fieldnames:
            if padrao in col.lower():
                return col

    for col in fieldnames:
        if col.lower() == "id":
            return col

    return None


def carregar_smiles_do_dataset(csv_path: Path):
    """
    Lê o dataset CLEAN e monta um dict {ligand_id: smiles}.
    Usa:
      - coluna de ID detectada automaticamente
      - coluna de SMILES = qualquer coluna que contenha 'smiles'
    """
    smiles_dict: dict[str, str] = {}

    if not csv_path.exists():
        print(f"[ERRO] Dataset CLEAN não encontrado: {csv_path}")
        sys.exit(1)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        col_smiles = next((col for col in fieldnames if "smiles" in col.lower()), None)
        if col_smiles is None:
            print("[ERRO] Nenhuma coluna contendo 'smiles' foi encontrada no dataset CLEAN.")
            print(f"Colunas disponíveis: {fieldnames}")
            sys.exit(1)

        col_id = detectar_coluna_id(fieldnames)
        if col_id is None:
            print("[ERRO] Não foi possível detectar uma coluna de ID adequada no dataset CLEAN.")
            print(f"Colunas disponíveis: {fieldnames}")
            sys.exit(1)

        print(f"[INFO] Usando coluna de ID: {col_id}")
        print(f"[INFO] Usando coluna de SMILES: {col_smiles}")

        for row in reader:
            lid_raw = row.get(col_id)
            smi_raw = row.get(col_smiles)
            if not lid_raw or not smi_raw:
                continue
            lid = str(lid_raw).strip()
            smi = smi_raw.strip()
            if lid and smi:
                smiles_dict[lid] = smi

    print(f"[INFO] Dicionário de SMILES carregado a partir do dataset CLEAN: {len(smiles_dict)} entradas.")
    return smiles_dict


def corrigir_topologia_e_estereo(pdb_path: Path, smiles_ref: str | None):
    """
    Combina coordenadas 3D do PDB com a topologia (ligações) do SMILES.
    """
    ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
    msg = ""
    method = ""

    try:
        # 1) Ler PDB cru
        raw_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if raw_mol is None:
            return None, "fail", "MolFromPDBFile retornou None", "no_pdb"

        mol_work = raw_mol

        # 2) Template SMILES (se disponível)
        if smiles_ref:
            template = Chem.MolFromSmiles(smiles_ref)
            if template:
                template = Chem.AddHs(template)
                Chem.AssignStereochemistry(template, cleanIt=True, force=True)

                try:
                    mol_work = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
                    method = "Template_SMILES"
                    msg = "Topologia corrigida via Template."
                except ValueError:
                    mol_work = raw_mol
                    method = "PDB_Fallback"
                    msg = "Falha no Template (átomos diferem). Usando PDB puro."
            else:
                method = "PDB_Fallback"
                msg = "SMILES inválido. Usando PDB puro."
        else:
            method = "PDB_NoSmiles"
            msg = "Sem SMILES. Usando PDB puro."

        # 3) Sanitização
        try:
            Chem.SanitizeMol(mol_work)
        except Exception as e:
            return None, "fail", f"Sanitização falhou: {e}", "sanitize_fail"

        # 4) Estereoquímica 3D
        if mol_work.GetNumConformers() > 0:
            try:
                try:
                    Chem.AssignStereochemistryFrom3D(
                        mol_work,
                        confId=-1,
                        replaceExistingTags=True
                    )
                    msg += " Estereo 3D aplicada (replaceExistingTags=True)."
                except TypeError:
                    Chem.AssignStereochemistryFrom3D(mol_work)
                    msg += " Estereo 3D aplicada (modo compatível)."
            except Exception:
                msg += " [AVISO] Falha ao ler Estereo 3D."

        n_atoms = mol_work.GetNumAtoms()
        msg += f" N_atoms={n_atoms}."

        mol_work.SetProp("_Name", ligand_id)

        return mol_work, "ok", msg, method

    except Exception as e:
        return None, "fail", f"Erro fatal: {e}", "exception"


# ============================================================================
# MAIN
# ============================================================================

def main():
    validar_diretorio_execucao()

    if not PDB_DIR.exists():
        print(f"[ERRO] Diretório {PDB_DIR} não existe.")
        sys.exit(1)

    pdb_files = sorted(PDB_DIR.glob("*_LIG1.pdb"))
    if not pdb_files:
        print(f"[ERRO] Nenhum arquivo PDB encontrado em {PDB_DIR}")
        sys.exit(1)

    print(f"[INFO] Processando {len(pdb_files)} arquivos PDB...")

    smiles_map = carregar_smiles_do_dataset(DATASET_CSV)

    writer = Chem.SDWriter(str(OUT_SDF))
    rows_log = []

    # Contadores para RELATÓRIO DE BATCH
    status_counts = Counter()
    method_counts = Counter()
    failed_ids = []

    sucessos = 0

    for pdb_path in pdb_files:
        ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
        smiles_ref = smiles_map.get(ligand_id)

        mol_final, status, msg, method = corrigir_topologia_e_estereo(pdb_path, smiles_ref)

        status_counts[status] += 1
        method_counts[method] += 1

        if mol_final and status == "ok":
            mol_final.SetProp("LigandID", ligand_id)
            mol_final.SetProp("Method", method)
            mol_final.SetProp("Source", "Boltz_Docking")
            if smiles_ref:
                mol_final.SetProp("Template_SMILES", smiles_ref)

            writer.write(mol_final)
            sucessos += 1
        else:
            failed_ids.append(ligand_id)

        rows_log.append({
            "ligand_id": ligand_id,
            "status": status,
            "method": method,
            "mensagem": msg
        })

        symbol = "✅" if status == "ok" else "❌"
        print(f"{symbol} [{ligand_id}] {method} -> {msg}")

    writer.close()

    # Salvar CSV de log
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["ligand_id", "status", "method", "mensagem"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_log)

    # =====================================================
    # RELATÓRIO RESUMIDO DE BATCH (para o terminal)
    # =====================================================
    print("-" * 60)
    print(f"[RESUMO] Processados: {len(pdb_files)} | Sucessos: {sucessos} | Falhas: {len(pdb_files) - sucessos}")
    print(f"[SAÍDA] SDF: {OUT_SDF}")
    print(f"[LOG]   CSV: {LOG_CSV}")
    print("-" * 60)
    print("[RELATÓRIO DE STATUS]")
    for st, n in status_counts.items():
        print(f"  - {st}: {n}")

    print("[RELATÓRIO DE MÉTODOS]")
    for m, n in method_counts.items():
        print(f"  - {m}: {n}")

    if method_counts.get("Template_SMILES", 0) == 0:
        print("[AVISO] Nenhum ligante conseguiu usar Template_SMILES. "
              "Todos caíram em PDB_Fallback/PDB_NoSmiles. "
              "Isso indica provável divergência entre o PDB do docking e o SMILES (contagem/ordem de átomos).")

    if failed_ids:
        print("[ATENÇÃO] Ligantes com falha (status != ok):")
        print("  " + ", ".join(failed_ids[:20]) + (" ..." if len(failed_ids) > 20 else ""))
    else:
        print("[INFO] Nenhum ligante com falha de processamento (status != ok).")


if __name__ == "__main__":
    main()
