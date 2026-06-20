from pymol import cmd
import csv
from pathlib import Path

# Diretório base é docking_boltz-2_V3 (um nível acima de scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent

MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
LOG_PATH = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1c_extrair_LIG1_pymol.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def escolher_ligante_menor(obj_name):
    """
    Seleciona o ligante candidato com base no tamanho (número de átomos),
    entre todos os orgânicos não-polímero e não-solvente.

    Retorna:
        (seleção_str, (chain, resi, resn), n_atoms) ou None se nada for encontrado.
    """
    sel_all = f"{obj_name} and organic and not polymer and not solvent"
    model = cmd.get_model(sel_all)

    if not model.atom:
        return None

    # Agrupar átomos por (cadeia, resi, resn)
    ligandos = {}  # (chain, resi, resn) -> contagem de átomos
    for at in model.atom:
        key = (at.chain, at.resi, at.resn)
        ligandos.setdefault(key, 0)
        ligandos[key] += 1

    # Escolher o ligante com MENOR número de átomos (esperado ser o inibidor, não o NADPH)
    best_key = None
    best_n = None
    for key, n in ligandos.items():
        if best_n is None or n < best_n:
            best_key = key
            best_n = n

    if best_key is None:
        return None

    chain, resi, resn = best_key

    # Construir uma seleção PyMOL para esse resíduo específico
    if chain:
        sel = f"({obj_name} and chain {chain} and resi {resi})"
    else:
        sel = f"({obj_name} and resi {resi})"

    return sel, best_key, best_n


def processar():
    cmd.reinitialize()

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

            # Escolher o ligante candidato pelo tamanho (menor número de átomos)
            result = escolher_ligante_menor(obj_name)
            if result is None:
                log.write(
                    f"[{ligand_id}] ERRO: nenhum ligante orgânico não-polímero encontrado.\n"
                )
                cmd.delete(obj_name)
                continue

            sel, (chain, resi, resn), n_atoms = result
            log.write(
                f"[{ligand_id}] Ligante candidato: chain='{chain}', resi='{resi}', "
                f"resn='{resn}', n_atoms={n_atoms}\n"
            )

            # Remover tudo que NÃO é o ligante candidato
            try:
                cmd.remove(f"not {sel}")
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao filtrar ligante candidato: {e}\n")
                cmd.delete(obj_name)
                continue

            out_path = OUT_DIR / f"{ligand_id}_LIG1.pdb"

            try:
                cmd.save(str(out_path), obj_name)
                log.write(f"[{ligand_id}] OK: LIG1 salvo em {out_path}\n")
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao salvar PDB: {e}\n")

            # Limpar objeto antes de passar para o próximo
            cmd.delete(obj_name)

    # Encerrar PyMOL após o processamento
    cmd.quit()


processar()
