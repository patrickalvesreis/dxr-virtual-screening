from pymol import cmd
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
LOG_PATH = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1c_extrair_LIG1_pymol.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def processar():
    with LOG_PATH.open("w", encoding="utf-8") as log:
        if not MAP_CSV.exists():
            log.write(f"[ERRO] Arquivo de mapeamento não encontrado: {MAP_CSV}\n")
            return

        with MAP_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            linhas = list(reader)

        log.write(f"Total de registros no mapa: {len(linhas)}\n\n")

        for linha in linhas:
            ligand_id = linha["ligand_id"]
            cif_path_str = linha.get("cif_model_0", "")
            cif_path = Path(cif_path_str) if cif_path_str else None

            if not cif_path or not cif_path.exists():
                log.write(f"[{ligand_id}] ERRO: cif_model_0 inexistente.\n")
                continue

            obj_name = f"complex_{ligand_id}"

            try:
                cmd.load(str(cif_path), obj_name)
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao carregar CIF: {e}\n")
                cmd.delete(obj_name)
                continue

            # Mantém só o resíduo do ligante candidato (assumindo resn LIG1)
            try:
                cmd.remove(f"not (resn LIG1 and model {obj_name})")
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao filtrar LIG1: {e}\n")
                cmd.delete(obj_name)
                continue

            out_path = OUT_DIR / f"{ligand_id}_LIG1.pdb"

            try:
                cmd.save(str(out_path), obj_name)
                log.write(f"[{ligand_id}] OK: LIG1 salvo em {out_path}\n")
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao salvar PDB: {e}\n")

            cmd.delete(obj_name)

    # Sai do PyMOL ao terminar
    cmd.quit()

processar()
