"""
SCRIPT: 10_debug_gnina.py
DESCRIÇÃO: Analisa os SDFs gerados. Se estiverem vazios, tenta rodar um teste verbose.
           Se tiverem conteúdo, tenta extrair scores com regex mais flexível.
"""

import os
import re
import subprocess
import pandas as pd

# --- CONFIGURAÇÕES ---
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "docking_data", "gnina_results")
GNINA_BIN = "/home/patrick/.local/bin/gnina"
RECEPTOR_PATH = os.path.join(PROJECT_DIR, "docking_data", "receptor.pdbqt")
LIGANDS_DIR = os.path.join(PROJECT_DIR, "docking_data", "ligands_pdbqt")

# Grid Box (Para teste de re-execução)
CENTER_X, CENTER_Y, CENTER_Z = 34.15, 13.80, 21.90
SIZE_X, SIZE_Y, SIZE_Z = 22, 22, 22

def parse_robust(sdf_path):
    """Tenta extrair scores de qualquer jeito."""
    scores = {}
    with open(sdf_path, 'r') as f:
        content = f.read()

    # Regex flexível para pegar números (inteiros ou floats)
    # Procura por > <TAG> \n VALOR

    # Vina Score / Minimized Affinity
    m = re.search(r'> <minimizedAffinity>[^-\d]*([-\d\.]+)', content)
    if m: scores['minimizedAffinity'] = float(m.group(1))

    # CNN Score
    m = re.search(r'> <CNNscore>[^-\d]*([-\d\.]+)', content)
    if m: scores['CNNscore'] = float(m.group(1))

    # CNN Affinity
    m = re.search(r'> <CNNaffinity>[^-\d]*([-\d\.]+)', content)
    if m: scores['CNNaffinity'] = float(m.group(1))

    return scores

def main():
    print("--- DEBUG RESULTADOS GNINA ---")

    sdf_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".sdf")]
    print(f"Encontrados {len(sdf_files)} arquivos SDF.")

    vazios = 0
    com_conteudo = 0
    sucesso_parse = 0

    results = []

    for sdf in sdf_files:
        path = os.path.join(OUTPUT_DIR, sdf)
        size = os.path.getsize(path)
        mol_id = sdf.replace("gnina_", "").replace(".sdf", "")

        if size == 0:
            vazios += 1
            if vazios == 1: # Testar apenas o primeiro erro
                print(f"\n[DEBUG] Arquivo {sdf} está vazio (0 bytes)!")
                print("Tentando rodar Gnina manualmente para ver o erro...")
                ligand_path = os.path.join(LIGANDS_DIR, f"ligand_{mol_id}.pdbqt")

                cmd = [
                    GNINA_BIN, "--receptor", RECEPTOR_PATH, "--ligand", ligand_path,
                    "--center_x", str(CENTER_X), "--center_y", str(CENTER_Y), "--center_z", str(CENTER_Z),
                    "--size_x", str(SIZE_X), "--size_y", str(SIZE_Y), "--size_z", str(SIZE_Z),
                    "--cpu", "4", "--num_modes", "1" # Rápido
                ]
                subprocess.run(cmd) # Roda com output visível no terminal
                print("-" * 30)
        else:
            com_conteudo += 1
            data = parse_robust(path)
            if data:
                sucesso_parse += 1
                row = {'ID': mol_id}
                row.update(data)
                results.append(row)
            else:
                # Se tem conteúdo mas não parseou, mostra o início do arquivo
                if sucesso_parse == 0:
                    print(f"\n[DEBUG] Conteúdo do {sdf} (não parseado):")
                    with open(path, 'r') as f:
                        print(f.read(500)) # Primeiros 500 chars
                    print("-" * 30)

    print(f"\nResumo:")
    print(f"Arquivos Vazios: {vazios}")
    print(f"Arquivos com Conteúdo: {com_conteudo}")
    print(f"Parseados com Sucesso: {sucesso_parse}")

    if sucesso_parse > 0:
        csv_path = os.path.join(PROJECT_DIR, "outputs", "gnina_debug_results.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Resultados recuperados salvos em: {csv_path}")

if __name__ == "__main__":
    main()
