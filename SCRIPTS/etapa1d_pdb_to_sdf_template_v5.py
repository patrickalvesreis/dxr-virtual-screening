#!/usr/bin/env python

from pathlib import Path
import csv
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================

# ONDE RODAR ESTE SCRIPT:
#   Na raiz do projeto:  docking_boltz-2_V3
BASE_DIR = Path.cwd()

PDB_DIR   = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
OUT_DIR   = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pdb_template"
LOG_CSV   = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1d_pdb_to_sdf_template_v4.csv"
DATASET_CLEAN = (BASE_DIR.parent / "dataset_dxr_v2_CLEAN.csv").resolve()
print(f"[DEBUG] Usando dataset CLEAN em: {DATASET_CLEAN}")

OUT_SDF   = OUT_DIR / "ligantes_corrigidos_v4.sdf"

# Garante diretórios
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def validar_diretorio_execucao():
    """Verifica se estamos na raiz correta do projeto."""
    if not (BASE_DIR / "01_Mapeamentos").exists():
        print("[ERRO CRÍTICO] Parece que você não está na raiz do projeto.")
        print(f"Diretório atual: {BASE_DIR}")
        print("Certifique-se de rodar o script a partir da pasta 'docking_boltz-2_V3'.")
        sys.exit(1)


def carregar_smiles_map(csv_path: Path):
    """
    Lê o dataset CLEAN e cria:
      - dict_smiles: {ligand_id (BindingDB MonomerID) -> SMILES}
    Detecta automaticamente a coluna de SMILES.
    """
    smiles_dict = {}

    if not csv_path.exists():
        print(f"[AVISO] Dataset CLEAN não encontrado: {csv_path}")
        return smiles_dict

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Coluna de ID
        col_id = None
        for col in fieldnames:
            if "bindingdb" in col.lower() and "monomer" in col.lower():
                col_id = col
                break
        if col_id is None:
            # fallback simples
            if "BindingDB MonomerID" in fieldnames:
                col_id = "BindingDB MonomerID"

        # Coluna de SMILES
        col_smiles = next((c for c in fieldnames if "smiles" in c.lower()), None)

        if col_id is None or col_smiles is None:
            print("[ERRO] Não foi possível detectar colunas de ID/SMILES no dataset CLEAN.")
            print(f"Fieldnames: {fieldnames}")
            return smiles_dict

        print(f"[INFO] Usando coluna de ID: {col_id}")
        print(f"[INFO] Usando coluna de SMILES: {col_smiles}")

        for row in reader:
            lid = row.get(col_id)
            smi = row.get(col_smiles)
            if lid and smi:
                smiles_dict[lid.strip()] = smi.strip()

    print(f"[INFO] Dicionário de SMILES carregado a partir do dataset CLEAN: {len(smiles_dict)} entradas.")
    return smiles_dict


def corrigir_topologia_e_estereo(pdb_path: Path, smiles_ref: str | None):
    """
    Versão 'Heavy Atom':
      - Remove Hs do PDB e do Template para garantir o match topológico.
      - Aplica AssignBondOrdersFromTemplate em átomos pesados.
      - Depois reinsere Hs corretamente, segundo a valência do SMILES.
    """
    ligand_id = pdb_path.name.replace("_LIG1.pdb", "")

    msg    = ""
    method = ""
    mol_final = None

    try:
        # 1) Ler PDB IGNORANDO Hidrogênios (removeHs=True)
        #    Isso evita falhas por diferenças de protonação entre Docking e RDKit.
        pdb_mol_no_h = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
        if pdb_mol_no_h is None:
            return None, "fail", "MolFromPDBFile falhou (None)", "no_pdb"

        # 2) Tentar aplicar Template via SMILES
        if smiles_ref:
            template = Chem.MolFromSmiles(smiles_ref)
            if template:
                # Remove Hs do template também, para comparar apenas átomos pesados
                template = Chem.RemoveHs(template)

                # Garante estereo básica no template
                Chem.AssignStereochemistry(template, cleanIt=True, force=True)

                try:
                    # Transferência de ordens de ligação considerando apenas heavy atoms
                    mol_with_bonds = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol_no_h)

                    # Agora adicionamos Hs de forma consistente com a valência do template
                    mol_final = Chem.AddHs(mol_with_bonds, addCoords=True)

                    method = "Template_SMILES"
                    msg = "Sucesso: Topologia via Template (match em átomos pesados)."
                except ValueError:
                    # Falha típica: contagem de heavy atoms diferente
                    method = "PDB_Fallback"
                    msg = "Falha no Template (átomos pesados diferem). Usando PDB original."
                    mol_final = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
            else:
                method = "PDB_Fallback"
                msg = "SMILES inválido. Usando PDB original."
                mol_final = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        else:
            method = "PDB_NoSmiles"
            msg = "Sem SMILES de referência. Usando PDB original."
            mol_final = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)

        if mol_final is None:
            return None, "fail", "mol_final None após processamento", method or "unknown"

        # 3) Sanitização final
        try:
            Chem.SanitizeMol(mol_final)
        except Exception as e:
            return None, "fail", f"Sanitização falhou: {e}", "sanitize_fail"

        # 4) Estereoquímica 3D (Refinamento a partir da pose docking)
        if mol_final.GetNumConformers() > 0:
            try:
                Chem.AssignStereochemistryFrom3D(
                    mol_final,
                    confId=-1,
                    replaceExistingTags=True
                )
                msg += " Estereo 3D OK."
            except Exception:
                msg += " [AVISO] Falha ao atribuir Estereo 3D."

        # Propriedades finais
        mol_final.SetProp("_Name", ligand_id)

        # Informação extra útil para debug
        n_atoms = mol_final.GetNumAtoms()
        msg += f" N_atoms={n_atoms}."

        return mol_final, "ok", msg, method

    except Exception as e:
        return None, "fail", f"Erro fatal: {e}", "exception"


# ==============================================================================
# MAIN
# ==============================================================================

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

    # Carrega mapa ligand_id -> SMILES do dataset CLEAN
    smiles_map = carregar_smiles_map(DATASET_CLEAN)

    writer = Chem.SDWriter(str(OUT_SDF))
    rows_log = []

    sucessos = 0
    contagem_metodos = {
        "Template_SMILES": 0,
        "PDB_Fallback": 0,
        "PDB_NoSmiles": 0,
        "sanitize_fail": 0,
        "no_pdb": 0,
        "exception": 0,
        "unknown": 0,
    }

    for pdb_path in pdb_files:
        ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
        smiles_ref = smiles_map.get(ligand_id)

        mol_final, status, msg, method = corrigir_topologia_e_estereo(pdb_path, smiles_ref)

        if method not in contagem_metodos:
            contagem_metodos["unknown"] += 1
        else:
            contagem_metodos[method] += 1

        if mol_final is not None and status == "ok":
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

        symbol = "✅" if status == "ok" else "❌"
        print(f"{symbol} [{ligand_id}] {method} -> {msg}")

    writer.close()

    # Salvar CSV de Log
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["ligand_id", "status", "method", "mensagem"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_log)

    print("-" * 60)
    print(f"[RESUMO] Processados: {len(pdb_files)} | Sucessos: {sucessos} | Falhas: {len(pdb_files) - sucessos}")
    print(f"[SAÍDA] SDF: {OUT_SDF}")
    print(f"[LOG]   CSV: {LOG_CSV}")
    print("-" * 60)
    print("[DETALHE MÉTODOS]")
    for m, c in contagem_metodos.items():
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
