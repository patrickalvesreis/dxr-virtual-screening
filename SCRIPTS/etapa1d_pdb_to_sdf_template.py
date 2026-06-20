from pathlib import Path
import csv
import sys

from rdkit import Chem
from rdkit.Chem import AllChem


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Como estamos rodando a partir da raiz do projeto (docking_boltz-2_V3)
BASE_DIR = Path.cwd()

PDB_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pdb_template"
LOG_CSV = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1d_pdb_to_sdf_template.csv"
MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"

OUT_SDF = OUT_DIR / "ligantes_LIG1_template.sdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def detectar_coluna_smiles(fieldnames):
    """
    Tenta detectar automaticamente a coluna de SMILES no CSV de mapeamento.
    Critério:
      - primeira coluna cujo nome contenha 'smiles' (case-insensitive).
    """
    if not fieldnames:
        return None

    for col in fieldnames:
        if "smiles" in col.lower():
            return col

    return None


def carregar_smiles_map(csv_path: Path):
    """
    Lê o CSV de mapeamento e cria um dicionário {ligand_id: smiles}.

    Requisitos:
      - coluna 'ligand_id'
      - alguma coluna de SMILES detectável via detectar_coluna_smiles()
    """
    smiles_dict = {}

    if not csv_path.exists():
        print(f"[ERRO] CSV de mapeamento não encontrado: {csv_path}")
        return smiles_dict

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print("[ERRO] CSV sem cabeçalho válido.")
            return smiles_dict

        if "ligand_id" not in fieldnames:
            print("[ERRO] CSV não contém coluna 'ligand_id'. Cabeçalhos encontrados:")
            print(fieldnames)
            return smiles_dict

        col_smiles = detectar_coluna_smiles(fieldnames)
        if not col_smiles:
            print("[ERRO] Não foi possível detectar coluna de SMILES.")
            print(f"Cabeçalhos disponíveis: {fieldnames}")
            return smiles_dict

        print(f"[INFO] Usando coluna de SMILES: '{col_smiles}'")

        for row in reader:
            lid = row.get("ligand_id")
            smi = row.get(col_smiles)
            if lid and smi:
                smiles_dict[lid] = smi

    print(f"[INFO] Total de entradas com SMILES carregadas: {len(smiles_dict)}")
    return smiles_dict


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    # 1) Verificar PDBs
    pdb_files = sorted(PDB_DIR.glob("*_LIG1.pdb"))
    if not pdb_files:
        print(f"[ERRO] Nenhum PDB *_LIG1.pdb encontrado em: {PDB_DIR}")
        sys.exit(1)

    print(f"[INFO] Encontrados {len(pdb_files)} PDBs em {PDB_DIR}")

    # 2) Carregar mapa ligand_id -> SMILES
    dict_smiles = carregar_smiles_map(MAP_CSV)
    if not dict_smiles:
        print("[ERRO] Dicionário de SMILES está vazio. Abortando.")
        sys.exit(1)

    # 3) Preparar escritor SDF e log
    writer_sdf = Chem.SDWriter(str(OUT_SDF))
    rows_log = []

    print("[INFO] Iniciando correção topológica via template SMILES...")

    for pdb_path in pdb_files:
        ligand_id = pdb_path.name.replace("_LIG1.pdb", "")
        smiles_ref = dict_smiles.get(ligand_id)

        status = "ok"
        msg = ""
        metodo = ""
        mol_final = None

        try:
            # A) Ler o PDB sem sanitizar (ligações ainda são "cruas")
            raw_mol = Chem.MolFromPDBFile(
                str(pdb_path),
                removeHs=False,
                sanitize=False
            )
            if raw_mol is None:
                raise ValueError("MolFromPDBFile retornou None")

            if raw_mol.GetNumAtoms() == 0:
                raise ValueError("Mol sem átomos")

            if smiles_ref:
                # B) Criar template perfeito a partir do SMILES
                template = Chem.MolFromSmiles(smiles_ref)
                if template is None:
                    msg = "SMILES inválido no CSV. Usando PDB com sanitização."
                    # Fallback: sanitizar o PDB e confiar na geometria
                    mol_tmp = Chem.Mol(raw_mol)
                    Chem.SanitizeMol(mol_tmp)
                    mol_final = mol_tmp
                    metodo = "PDB_sanitize_no_template"
                else:
                    # Opcional: adicionar Hs ao template para compatibilidade
                    template_H = Chem.AddHs(template)

                    try:
                        # C) Transferir ordens de ligação do template para o PDB
                        mol_template = AllChem.AssignBondOrdersFromTemplate(
                            template_H,
                            raw_mol
                        )
                        # Sanitizar para garantir consistência química
                        Chem.SanitizeMol(mol_template)
                        mol_final = mol_template
                        metodo = "Template_SMILES"
                        msg = "Topologia corrigida via AssignBondOrdersFromTemplate"
                    except Exception as e:
                        # D) Fallback se o template não encaixar (nº de átomos etc.)
                        msg = (
                            "Falha no AssignBondOrdersFromTemplate "
                            f"(usando PDB sanitizado). Detalhe: {e}"
                        )
                        mol_tmp = Chem.Mol(raw_mol)
                        Chem.SanitizeMol(mol_tmp)
                        mol_final = mol_tmp
                        metodo = "PDB_sanitize_template_fail"
            else:
                # E) Sem SMILES de referência: confiar no PDB sanitizado
                msg = "Sem SMILES de referência. Usando PDB sanitizado."
                mol_tmp = Chem.Mol(raw_mol)
                Chem.SanitizeMol(mol_tmp)
                mol_final = mol_tmp
                metodo = "PDB_sanitize_no_smiles"

            if mol_final is None:
                raise ValueError("Mol_final é None após o processamento.")

            # Atribuir propriedades para rastreabilidade
            mol_final.SetProp("_Name", ligand_id)
            mol_final.SetProp("LigandID", ligand_id)
            mol_final.SetProp("SourcePDB", str(pdb_path))
            mol_final.SetProp("Method", metodo)
            if msg:
                mol_final.SetProp("Note", msg)

            writer_sdf.write(mol_final)

        except Exception as e:
            status = "fail"
            msg = str(e)
            metodo = "error"
            print(f"[FALHA] {ligand_id}: {msg}")

        rows_log.append(
            {
                "ligand_id": ligand_id,
                "pdb_path": str(pdb_path),
                "status": status,
                "method": metodo,
                "message": msg,
            }
        )

    writer_sdf.close()

    # 4) Salvar log CSV
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=["ligand_id", "pdb_path", "status", "method", "message"],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows_log)

    total = len(rows_log)
    ok = sum(1 for r in rows_log if r["status"] == "ok")
    fail = total - ok

    print("-" * 60)
    print(f"[RESUMO] Total: {total} | OK: {ok} | Falhas: {fail}")
    print(f"[INFO] SDF corrigido salvo em: {OUT_SDF}")
    print(f"[INFO] Log detalhado salvo em: {LOG_CSV}")


if __name__ == "__main__":
    main()
