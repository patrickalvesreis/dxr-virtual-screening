#!/usr/bin/env python3
"""
Etapa 1B – Extrair ligantes 3D a partir de mol_pkl do Boltz-2

Objetivo:
    - Ler 01_Mapeamentos/map_boltz2_outputs.csv
    - Para cada ligand_id com mol_pkl:
        * carregar o pickle
        * extrair TODOS os RDKit Mol (LIG1, LIG2, etc.)
        * salvar em:
              02_Ligantes_3D/sdf_from_pkl/<lig_key>/<ligand_id>_<lig_key>.sdf
    - Registrar sucessos e falhas em:
        * 02_Ligantes_3D/logs/etapa1b_extrair_ligantes_3d_pkl.log
        * 02_Ligantes_3D/logs/etapa1b_resumo_extracao_pkl.csv
"""

import csv
import pickle
from pathlib import Path

from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent.parent
MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"

OUT_BASE_SDF_DIR = BASE_DIR / "02_Ligantes_3D" / "sdf_from_pkl"
OUT_BASE_SDF_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "02_Ligantes_3D" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_TXT = LOG_DIR / "etapa1b_extrair_ligantes_3d_pkl.log"
SUMMARY_CSV = LOG_DIR / "etapa1b_resumo_extracao_pkl.csv"


def extrair_mols_rdkit(obj):
    """
    Tenta extrair TODOS os RDKit Mol de um objeto genérico vindo do pickle.

    Retorna:
        dict[str, Chem.Mol], por exemplo:
            {"LIG1": Mol(...), "LIG2": Mol(...)}
    """
    mols = {}

    # Caso direto: um único Mol
    if isinstance(obj, Chem.Mol):
        mols["MOL"] = obj
        return mols

    # Caso dict: procurar valores que sejam Mol
    if isinstance(obj, dict):
        for key, v in obj.items():
            if isinstance(v, Chem.Mol):
                mols[str(key)] = v
        return mols

    # Caso lista/tupla: indexar mols encontrados
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, Chem.Mol):
                mols[f"mol_{i}"] = v
        return mols

    # Caso não previsto
    return mols


def main():
    if not MAP_CSV.exists():
        raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {MAP_CSV}")

    with MAP_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        linhas = list(reader)

    total = len(linhas)
    total_registros_mol = 0
    falhas = 0

    summary_rows = []

    with LOG_TXT.open("w", encoding="utf-8") as log:
        log.write("Etapa 1B – Extração de ligantes 3D a partir de mol_pkl\n")
        log.write(f"Total de registros no mapa: {total}\n\n")

        for linha in linhas:
            ligand_id = linha["ligand_id"]
            mol_pkl_path_str = linha.get("mol_pkl", "")
            mol_pkl_path = Path(mol_pkl_path_str) if mol_pkl_path_str else None

            if not mol_pkl_path or not mol_pkl_path.exists():
                status = "SEM_MOL_PKL"
                msg = "Arquivo mol_pkl ausente ou caminho vazio."
                falhas += 1
                log.write(f"[{ligand_id}] {status}: {msg}\n")
                summary_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "lig_key": "",
                        "status": status,
                        "mensagem": msg,
                        "sdf_path": "",
                        "num_conformers": "",
                        "num_atoms": "",
                    }
                )
                continue

            # Carregar o pickle
            try:
                with mol_pkl_path.open("rb") as pf:
                    obj = pickle.load(pf)
            except Exception as e:
                status = "ERRO_PICKLE"
                msg = f"Falha ao carregar pickle: {e}"
                falhas += 1
                log.write(f"[{ligand_id}] {status}: {msg}\n")
                summary_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "lig_key": "",
                        "status": status,
                        "mensagem": msg,
                        "sdf_path": "",
                        "num_conformers": "",
                        "num_atoms": "",
                    }
                )
                continue

            mols_dict = extrair_mols_rdkit(obj)

            if not mols_dict:
                status = "SEM_MOL_RDKit"
                msg = "Nenhum RDKit Mol encontrado no objeto carregado."
                falhas += 1
                log.write(
                    f"[{ligand_id}] {status}: {msg} (type(obj) = {type(obj)})\n"
                )
                summary_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "lig_key": "",
                        "status": status,
                        "mensagem": msg,
                        "sdf_path": "",
                        "num_conformers": "",
                        "num_atoms": "",
                    }
                )
                continue

            # Para cada Mol encontrado (LIG1, LIG2, etc.)
            for lig_key, mol in mols_dict.items():
                num_conf = mol.GetNumConformers()
                num_atoms = mol.GetNumAtoms()

                # Diretório para essa "família" de ligante (LIG1, LIG2, ...)
                lig_dir = OUT_BASE_SDF_DIR / lig_key
                lig_dir.mkdir(parents=True, exist_ok=True)

                sdf_path = lig_dir / f"{ligand_id}_{lig_key}.sdf"

                try:
                    w = Chem.SDWriter(str(sdf_path))
                    w.write(mol)
                    w.close()
                except Exception as e:
                    status = "ERRO_SALVAR_SDF"
                    msg = f"Erro ao salvar SDF para {lig_key}: {e}"
                    falhas += 1
                    log.write(
                        f"[{ligand_id}::{lig_key}] {status}: {msg}\n"
                    )
                    summary_rows.append(
                        {
                            "ligand_id": ligand_id,
                            "lig_key": lig_key,
                            "status": status,
                            "mensagem": msg,
                            "sdf_path": "",
                            "num_conformers": "",
                            "num_atoms": "",
                        }
                    )
                    continue

                total_registros_mol += 1
                status = "OK"
                msg = ""
                log.write(
                    f"[{ligand_id}::{lig_key}] OK: SDF salvo em {sdf_path} "
                    f"(conformers={num_conf}, atoms={num_atoms})\n"
                )
                summary_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "lig_key": lig_key,
                        "status": status,
                        "mensagem": msg,
                        "sdf_path": str(sdf_path),
                        "num_conformers": num_conf,
                        "num_atoms": num_atoms,
                    }
                )

        log.write("\nResumo:\n")
        log.write(f"  Registros de Mol salvos (ligand_id x lig_key): {total_registros_mol}\n")
        log.write(f"  Falhas: {falhas}\n")

    # Salvar CSV de resumo
    fieldnames = [
        "ligand_id",
        "lig_key",
        "status",
        "mensagem",
        "sdf_path",
        "num_conformers",
        "num_atoms",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Extração concluída.")
    print(f"     Registros de Mol salvos: {total_registros_mol}")
    print(f"     Falhas:                 {falhas}")
    print(f"Log detalhado: {LOG_TXT}")
    print(f"Resumo CSV:    {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
