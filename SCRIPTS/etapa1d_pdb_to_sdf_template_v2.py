from pathlib import Path
import csv
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# ONDE RODAR ESTE SCRIPT:
#   cd ~/1_PROJETO_DOCAGEM/Projeto_DXR/Banco-de-dados-bindingDB/Descritores/PROJETO_DXR_ML_GNN/00_Dados_Originais/docking_boltz-2_V3
#   (chem_gnn) python scripts/etapa1d_pdb_to_sdf_template_v2.py

BASE_DIR = Path.cwd()

PDB_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pdb_template"
LOG_CSV = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1d_pdb_to_sdf_template_v2.csv"
MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"
OUT_SDF = OUT_DIR / "ligantes_corrigidos_v2.sdf"

# Garante diretórios
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def validar_diretorio_execucao():
    """Verifica se estamos na raiz esperada do projeto."""
    if not (BASE_DIR / "01_Mapeamentos").exists():
        print("[ERRO CRÍTICO] Parece que você não está na raiz do projeto.")
        print(f"Diretório atual: {BASE_DIR}")
        print("Certifique-se de rodar o script a partir da pasta 'docking_boltz-2_V3'.")
        sys.exit(1)


def carregar_smiles_map(csv_path: Path):
    """
    Lê o CSV de mapeamento e tenta encontrar uma coluna de SMILES.
    Retorna: dict {ligand_id: smiles}
    """
    smiles_dict: dict[str, str] = {}

    if not csv_path.exists():
        print(f"[AVISO] CSV de mapeamento não encontrado: {csv_path}")
        return smiles_dict

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Detecção automática de coluna (case-insensitive)
        col_smiles = next((col for col in fieldnames if "smiles" in col.lower()), None)

        if col_smiles is None:
            print("[ERRO] Nenhuma coluna contendo 'smiles' foi encontrada no CSV.")
            return smiles_dict

        for row in reader:
            lid = row.get("ligand_id")
            smi = row.get(col_smiles)
            if lid and smi:
                smiles_dict[lid] = smi.strip()

    print(f"[INFO] Dicionário de SMILES carregado: {len(smiles_dict)} entradas.")
    return smiles_dict


def corrigir_topologia_e_estereo(pdb_path: Path, smiles_ref: str | None):
    """
    Combina coordenadas 3D do PDB com a topologia (ligações) do SMILES.
    Passos:
      1) Lê o PDB sem sanitizar (mantém Hs e evita erros precoce).
      2) Se houver SMILES, usa como template em AssignBondOrdersFromTemplate.
      3) Sanitiza a molécula resultante.
      4) Atribui estereoquímica a partir da geometria 3D.
    """
    ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
    msg = ""
    method = ""

    try:
        # 1) Ler PDB cru (sem sanitizar)
        raw_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if raw_mol is None:
            return None, "fail", "MolFromPDBFile retornou None", "no_pdb"

        mol_work = raw_mol

        # 2) Tentar aplicar Template (SMILES como “verdade química”)
        if smiles_ref:
            template = Chem.MolFromSmiles(smiles_ref)
            if template:
                # Adiciona Hs ao template para alinhar com o PDB (que tem H explícito)
                template = Chem.AddHs(template)

                # Garantir estereoquímica básica no template
                Chem.AssignStereochemistry(template, cleanIt=True, force=True)

                try:
                    # Transfere a conectividade do SMILES para os átomos do PDB
                    mol_work = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
                    method = "Template_SMILES"
                    msg = "Topologia corrigida via Template."
                except ValueError:
                    # Falha típica: número / ordem de átomos diverge
                    mol_work = raw_mol
                    method = "PDB_Fallback"
                    msg = "Falha no Template (átomos diferem). Usando PDB puro."
            else:
                method = "PDB_Fallback"
                msg = "SMILES inválido. Usando PDB puro."
        else:
            method = "PDB_NoSmiles"
            msg = "Sem SMILES. Usando PDB puro."

        # 3) Sanitização (validação química)
        try:
            Chem.SanitizeMol(mol_work)
        except Exception as e:
            return None, "fail", f"Sanitização falhou: {e}", "sanitize_fail"

        # 4) Estereoquímica 3D (deixa a pose mandar nos tags R/S, E/Z)
        if mol_work.GetNumConformers() > 0:
            try:
                # Alguns RDKit antigos podem não aceitar replaceExistingTags;
                # fazemos fallback elegante se isso acontecer.
                try:
                    Chem.AssignStereochemistryFrom3D(
                        mol_work,
                        confId=-1,
                        replaceExistingTags=True
                    )
                    msg += " Estereo 3D aplicada (replaceExistingTags=True)."
                except TypeError:
                    Chem.AssignStereochemistryFrom3D(mol_work)
                    msg += " Estereo 3D aplicada (sem replaceExistingTags: compatibilidade)."
            except Exception:
                msg += " [AVISO] Falha ao ler Estereo 3D."

        # 5) Metadado básico: número de átomos
        n_atoms = mol_work.GetNumAtoms()
        msg += f" N_atoms={n_atoms}."

        # Nome amigável no SDF
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

    smiles_map = carregar_smiles_map(MAP_CSV)
    writer = Chem.SDWriter(str(OUT_SDF))
    rows_log = []

    sucessos = 0

    for pdb_path in pdb_files:
        ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
        smiles_ref = smiles_map.get(ligand_id)

        mol_final, status, msg, method = corrigir_topologia_e_estereo(pdb_path, smiles_ref)

        if mol_final and status == "ok":
            # Metadados para rastreabilidade
            mol_final.SetProp("LigandID", ligand_id)
            mol_final.SetProp("Method", method)
            mol_final.SetProp("Source", "Boltz_Docking")
            if smiles_ref:
                mol_final.SetProp("Template_SMILES", smiles_ref)

            writer.write(mol_final)
            sucessos += 1

        rows_log.append({
            "ligand_id": ligand_id,
            "status": status,
            "method": method,
            "mensagem": msg
        })

        # Log visual minimalista no terminal
        symbol = "✅" if status == "ok" else "❌"
        print(f"{symbol} [{ligand_id}] {method} -> {msg}")

    writer.close()

    # Salvar CSV de log
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["ligand_id", "status", "method", "mensagem"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_log)

    print("-" * 60)
    print(f"[RESUMO] Processados: {len(pdb_files)} | Sucessos: {sucessos} | Falhas: {len(pdb_files) - sucessos}")
    print(f"[SAÍDA] SDF: {OUT_SDF}")
    print(f"[LOG]   CSV: {LOG_CSV}")


if __name__ == "__main__":
    main()
