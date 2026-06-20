"""
SCRIPT: 09_rodar_gnina_docking_final.py
AMBIENTE: 'micromamba activate docking'
DESCRIÇÃO: Docking com Gnina.
MELHORIA: Regex robusto (aceita notação científica e espaços variados) e sistema de recuperação de falhas.
"""

import os
import subprocess
import pandas as pd
import time
import re

# --- CONFIGURAÇÕES ---
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"

# Caminhos
GNINA_BIN = "/home/patrick/.local/bin/gnina"
RECEPTOR_PATH = os.path.join(PROJECT_DIR, "docking_data", "receptor.pdbqt")
LIGANDS_DIR = os.path.join(PROJECT_DIR, "docking_data", "ligands_pdbqt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "docking_data", "gnina_results")
RESULTS_CSV = os.path.join(PROJECT_DIR, "outputs", "resultados_docking_gnina.csv")

# --- GRID BOX CORRIGIDO ---
CENTER_X = 5.36
CENTER_Y = 0.58
CENTER_Z = 44.27
SIZE_X = 22
SIZE_Y = 22
SIZE_Z = 22

# Configurações Gnina
CPU_CORES = 4
NUM_MODES = 9

def parse_gnina_sdf(sdf_file):
    """Extrai scores do SDF com REGEX ROBUSTO (aceita espaços e notação científica)."""
    scores = {"minimizedAffinity": None, "CNNscore": None, "CNNaffinity": None}
    try:
        with open(sdf_file, 'r') as f:
            content = f.read()

        # Pega a primeira pose (a melhor)
        poses = content.split('$$$$')
        if len(poses) > 0:
            best_pose = poses[0]

            # --- REGEX BLINDADO ---
            # \s* = zero ou mais espaços/quebras de linha
            # [-\d\.Ee+]+ = captura números, negativos, decimais e notação científica (ex: 1.2e-5)

            m_aff = re.search(r'>\s*<minimizedAffinity>\s*([-\d\.Ee+]+)', best_pose)
            if m_aff: scores["minimizedAffinity"] = float(m_aff.group(1))

            m_cnn_s = re.search(r'>\s*<CNNscore>\s*([-\d\.Ee+]+)', best_pose)
            if m_cnn_s: scores["CNNscore"] = float(m_cnn_s.group(1))

            m_cnn_a = re.search(r'>\s*<CNNaffinity>\s*([-\d\.Ee+]+)', best_pose)
            if m_cnn_a: scores["CNNaffinity"] = float(m_cnn_a.group(1))

    except Exception as e:
        # Erro silencioso na leitura para não travar o loop, mas registrado se necessário
        pass
    return scores

def main():
    print("--- DOCKING GNINA: EXTRAÇÃO E EXECUÇÃO (VERSÃO FINAL) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ligand_files = [f for f in os.listdir(LIGANDS_DIR) if f.endswith(".pdbqt")]
    total = len(ligand_files)
    print(f"Total de ligantes na pasta: {total}")

    results = []
    recuperados = 0
    processados = 0

    start_time = time.time()

    for i, lig_file in enumerate(ligand_files):
        mol_id = lig_file.replace("ligand_", "").replace(".pdbqt", "")
        input_ligand = os.path.join(LIGANDS_DIR, lig_file)
        output_sdf = os.path.join(OUTPUT_DIR, f"gnina_{mol_id}.sdf")

        # 1. Verifica se o SDF já existe e tem conteúdo
        ja_existe = os.path.exists(output_sdf) and os.path.getsize(output_sdf) > 0

        if not ja_existe:
            # Roda o GNINA apenas se não existir
            status = "NOVO"
            cmd = [
                GNINA_BIN,
                "--receptor", RECEPTOR_PATH,
                "--ligand", input_ligand,
                "--out", output_sdf,
                "--center_x", str(CENTER_X), "--center_y", str(CENTER_Y), "--center_z", str(CENTER_Z),
                "--size_x", str(SIZE_X), "--size_y", str(SIZE_Y), "--size_z", str(SIZE_Z),
                "--cpu", str(CPU_CORES), "--num_modes", str(NUM_MODES)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                processados += 1
            except subprocess.CalledProcessError:
                print(f"[{i+1}/{total}] ID {mol_id}: Erro CRÍTICO na execução do Gnina.")
                continue
        else:
            status = "RECUPERADO"
            recuperados += 1

        # 2. Extrai os dados
        data = parse_gnina_sdf(output_sdf)

        if data["minimizedAffinity"] is not None:
            row = {"ID": mol_id}
            row.update(data)
            results.append(row)
            # Log limpo
            print(f"[{i+1}/{total}] {status} ID {mol_id} | CNN Affinity: {data['CNNaffinity']}")
        else:
            print(f"[{i+1}/{total}] {status} ID {mol_id}: ARQUIVO EXISTE MAS SEM DADOS LEITURA.")

    # Salvar resultados
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_CSV, index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"\n--- RELATÓRIO FINAL ---")
    print(f"Tempo total: {elapsed:.2f} minutos")
    print(f"Novos dockings executados: {processados}")
    print(f"Arquivos recuperados do disco: {recuperados}")
    print(f"Total de moléculas com sucesso: {len(results)}")
    print(f"Tabela salva em: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
