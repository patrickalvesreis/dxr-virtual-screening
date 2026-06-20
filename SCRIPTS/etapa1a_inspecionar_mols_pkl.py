#!/usr/bin/env python3
"""
Etapa 1A – Inspecionar arquivos mol_pkl do Boltz-2

Objetivo:
    - Carregar alguns arquivos .pkl de mols do Boltz-2
    - Ver qual é o tipo de objeto salvo (RDKit Mol, dict, lista, etc.)
    - Registrar essa inspeção em um log para ajustar a extração 3D na Etapa 1B.

Uso:
    python scripts/etapa1a_inspecionar_mols_pkl.py
"""

import csv
import pickle
from pathlib import Path

try:
    from rdkit import Chem  # só para checar se o objeto é um Mol
except ImportError:
    Chem = None

BASE_DIR = Path(__file__).resolve().parent.parent
MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"
LOG_PATH = BASE_DIR / "02_Ligantes_3D" / "logs" / "etapa1a_inspecionar_mols_pkl.log"


def main():
    if not MAP_CSV.exists():
        raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {MAP_CSV}")

    with MAP_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        linhas = list(reader)

    # Filtrar entradas com mol_pkl
    linhas_mol = [l for l in linhas if l.get("mol_pkl")]
    print(f"Total de entradas com mol_pkl: {len(linhas_mol)}")

    # Vamos inspecionar só os primeiros N
    N = min(5, len(linhas_mol))
    print(f"Inspecionando os primeiros {N} arquivos mol_pkl...")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write("Etapa 1A – Inspeção de mol_pkl\n")
        log.write(f"Total com mol_pkl: {len(linhas_mol)}\n\n")

        for linha in linhas_mol[:N]:
            ligand_id = linha["ligand_id"]
            mol_pkl_path = Path(linha["mol_pkl"])

            log.write(f"=== ligand_id: {ligand_id} ===\n")
            log.write(f"mol_pkl_path: {mol_pkl_path}\n")

            if not mol_pkl_path.exists():
                log.write("  [ERRO] Arquivo mol_pkl não encontrado.\n\n")
                continue

            try:
                with mol_pkl_path.open("rb") as pf:
                    obj = pickle.load(pf)
            except Exception as e:
                log.write(f"  [ERRO] Falha ao carregar pickle: {e}\n\n")
                continue

            # Tipo bruto do objeto
            log.write(f"  type(obj): {type(obj)}\n")

            # Se for RDKit Mol
            if Chem is not None and isinstance(obj, Chem.Mol):
                log.write("  Detecção: objeto é RDKit Chem.Mol\n")
                log.write(f"  Num átomos: {obj.GetNumAtoms()}\n")
                log.write(f"  Num conformers: {obj.GetNumConformers()}\n\n")
                continue

            # Se for dict
            if isinstance(obj, dict):
                log.write("  Detecção: objeto é dict\n")
                log.write(f"  Keys: {list(obj.keys())}\n")
                # Tentar achar algo que pareça um Mol
                if Chem is not None:
                    for k, v in obj.items():
                        if isinstance(v, Chem.Mol):
                            log.write(f"  -> Possível Mol em key '{k}': {v.GetNumAtoms()} átomos, "
                                      f"{v.GetNumConformers()} conformers\n")
                log.write("\n")
                continue

            # Se for lista ou tupla
            if isinstance(obj, (list, tuple)):
                log.write(f"  Detecção: objeto é {type(obj)} com len={len(obj)}\n")
                if len(obj) > 0:
                    primeiro = obj[0]
                    log.write(f"  type(obj[0]): {type(primeiro)}\n")
                    if Chem is not None and isinstance(primeiro, Chem.Mol):
                        log.write("  -> obj[0] é um RDKit Mol\n")
                        log.write(f"     Num átomos: {primeiro.GetNumAtoms()}\n")
                        log.write(f"     Num conformers: {primeiro.GetNumConformers()}\n")
                log.write("\n")
                continue

            # Caso genérico
            log.write("  [INFO] Objeto em formato não previsto (nem Mol, nem dict, nem list/tuple)\n\n")

    print(f"[OK] Log de inspeção salvo em: {LOG_PATH}")


if __name__ == "__main__":
    main()
