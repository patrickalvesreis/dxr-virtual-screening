#!/usr/bin/env python3
"""
Etapa 0 – Mapear saídas do Boltz-2 para o projeto DXR

Objetivo:
    - Para cada ID de ligante (ex.: 50028309), registrar:
        * caminho do YAML de entrada
        * caminhos dos arquivos de previsão (cif, affinity, plddt, pae, pde, pre_affinity)
        * caminhos dos arquivos processados (mol.pkl, records, structures, constraints)
    - Gerar um CSV em 01_Mapeamentos/map_boltz2_outputs.csv

Uso:
    python scripts/etapa0_mapear_boltz2.py
"""

import csv
from pathlib import Path

# Base = diretório docking_boltz-2_V3 (onde estão inputs_yaml e resultados_fp32_seq)
BASE_DIR = Path(__file__).resolve().parent.parent

INPUTS_YAML_DIR = BASE_DIR / "inputs_yaml"
RESULTS_DIR = BASE_DIR / "resultados_fp32_seq"
OUT_DIR = BASE_DIR / "01_Mapeamentos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "map_boltz2_outputs.csv"


def mapear_boltz2():
    registros = []

    if not INPUTS_YAML_DIR.exists():
        raise FileNotFoundError(f"Diretório inputs_yaml não encontrado: {INPUTS_YAML_DIR}")

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Diretório resultados_fp32_seq não encontrado: {RESULTS_DIR}")

    # IDs de ligantes a partir dos YAMLs
    yaml_ids = {p.stem for p in INPUTS_YAML_DIR.glob("*.yaml")}

    # IDs de ligantes a partir dos diretórios de resultados
    result_ids = {p.name for p in RESULTS_DIR.iterdir() if p.is_dir()}

    # Conjunto total de IDs
    all_ids = sorted(yaml_ids.union(result_ids), key=lambda x: (len(x), x))

    print(f"Total IDs em inputs_yaml: {len(yaml_ids)}")
    print(f"Total IDs em resultados_fp32_seq: {len(result_ids)}")
    print(f"Total IDs únicos: {len(all_ids)}")

    for ligand_id in all_ids:
        # Caminho do YAML
        yaml_path = INPUTS_YAML_DIR / f"{ligand_id}.yaml"
        yaml_path_str = str(yaml_path) if yaml_path.exists() else ""

        # Diretório principal de resultados desse ID
        run_dir = RESULTS_DIR / ligand_id
        run_dir_str = str(run_dir) if run_dir.exists() else ""

        boltz_dir = run_dir / f"boltz_results_{ligand_id}"

        # predictions/<ID>/
        pred_dir = boltz_dir / "predictions" / ligand_id

        cif_model_0 = pred_dir / f"{ligand_id}_model_0.cif"
        affinity_json = pred_dir / f"affinity_{ligand_id}.json"
        confidence_json = pred_dir / f"confidence_{ligand_id}_model_0.json"
        pae_npz = pred_dir / f"pae_{ligand_id}_model_0.npz"
        pde_npz = pred_dir / f"pde_{ligand_id}_model_0.npz"
        plddt_npz = pred_dir / f"plddt_{ligand_id}_model_0.npz"
        pre_affinity_npz = pred_dir / f"pre_affinity_{ligand_id}.npz"

        # processed/
        proc_dir = boltz_dir / "processed"
        constraints_npz = proc_dir / "constraints" / f"{ligand_id}.npz"
        manifest_json = proc_dir / "manifest.json"
        mol_pkl = proc_dir / "mols" / f"{ligand_id}.pkl"
        msa_npz = proc_dir / "msa" / f"{ligand_id}_0.npz"
        record_json = proc_dir / "records" / f"{ligand_id}.json"
        structures_npz = proc_dir / "structures" / f"{ligand_id}.npz"

        def ok(path: Path) -> str:
            return str(path) if path.exists() else ""

        registro = {
            "ligand_id": ligand_id,
            "yaml_path": yaml_path_str,
            "run_dir": run_dir_str,
            "boltz_dir": ok(boltz_dir),

            "cif_model_0": ok(cif_model_0),
            "affinity_json": ok(affinity_json),
            "confidence_json": ok(confidence_json),
            "pae_npz": ok(pae_npz),
            "pde_npz": ok(pde_npz),
            "plddt_npz": ok(plddt_npz),
            "pre_affinity_npz": ok(pre_affinity_npz),

            "constraints_npz": ok(constraints_npz),
            "manifest_json": ok(manifest_json),
            "mol_pkl": ok(mol_pkl),
            "msa_npz": ok(msa_npz),
            "record_json": ok(record_json),
            "structures_npz": ok(structures_npz),

            # flags para sanity check
            "has_yaml": yaml_path.exists(),
            "has_results_dir": run_dir.exists(),
            "has_cif": cif_model_0.exists(),
            "has_affinity": affinity_json.exists(),
            "has_mol_pkl": mol_pkl.exists(),
        }

        registros.append(registro)

    campos = list(registros[0].keys()) if registros else []

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)

    print(f"[OK] Mapeamento salvo em: {OUT_CSV}")
    print(f"Total de registros escritos: {len(registros)}")


if __name__ == "__main__":
    mapear_boltz2()
