"""
SCRIPT: 13_preparar_visualizacao_pymol_lig_50181153.py
AMBIENTE: 'micromamba activate viz'
DESCRIÇÃO: Gera o script .pml e abre o PyMOL.
CORREÇÃO: Resolve o erro UnboundLocalError definindo caminhos localmente.
"""

import os
import subprocess
import sys

# --- CONFIGURAÇÕES GERAIS ---
PROJECT_DIR = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase"
LIGAND_ID = "50181153"

def main():
    print(f"--- PREPARANDO CENA PYMOL: LIGANTE {LIGAND_ID} ---")

    # 1. Definir Caminhos (Localmente para evitar erro de escopo)
    receptor_path = os.path.join(PROJECT_DIR, "docking_data", "receptor.pdbqt")
    output_pml = os.path.join(PROJECT_DIR, "outputs", "ver_campeao.pml")

    # Caminho padrão do ligante
    ligand_path = os.path.join(PROJECT_DIR, "docking_data", "gnina_results", f"gnina_{LIGAND_ID}.sdf")

    # 2. Verificar Arquivos
    if not os.path.exists(receptor_path):
        print(f"ERRO: Receptor não encontrado: {receptor_path}")
        return

    # Lógica de verificação do ligante
    if not os.path.exists(ligand_path):
        print(f"Aviso: Não achei {ligand_path}")
        # Tenta achar sem o prefixo gnina_
        alt_path = os.path.join(PROJECT_DIR, "docking_data", "gnina_results", f"{LIGAND_ID}.sdf")
        if os.path.exists(alt_path):
            print(f"-> Achei com nome alternativo: {alt_path}")
            ligand_path = alt_path
        else:
            print("ERRO FATAL: Ligante não encontrado na pasta gnina_results.")
            return
    else:
        print(f"-> Ligante encontrado: {ligand_path}")

    # 3. Conteúdo do Script PyMOL (.pml)
    pml_content = f"""
# --- SCRIPT AUTOMÁTICO PARA PYMOL ---
reinitialize

# A. Carregar Estruturas
load {receptor_path}, receptor
load {ligand_path}, ligante

# B. Estilo da Proteína
bg_color white
hide everything
show cartoon, receptor
color white, receptor
set cartoon_transparency, 0.3

# C. Estilo do Ligante
show sticks, ligante
color green, ligante
util.cbag ligante

# D. Definir o "Bolso" (Pocket)
select pocket, (receptor within 5 of ligante)
show sticks, pocket
color cyan, pocket
util.cbac pocket
deselect

# E. Interações Polares
dist interacoes, ligante, pocket, mode=2
color magenta, interacoes
set dash_gap, 0.3
set dash_width, 3.0
set dash_radius, 0.1

# F. Labels
label n. CA and pocket, "%s-%s" % (resn, resi)
set label_color, black
set label_size, 14
set label_position, (0, 0, 10)

# G. Zoom final
zoom ligante, 8
    """

    # 4. Salvar arquivo .pml
    try:
        with open(output_pml, "w") as f:
            f.write(pml_content)
        print(f"Script PML gerado em: {output_pml}")
    except Exception as e:
        print(f"Erro ao salvar PML: {e}")
        return

    # 5. Abrir o PyMOL Automaticamente
    print("--- INICIANDO PYMOL ---")
    try:
        subprocess.run(["pymol", output_pml])
    except FileNotFoundError:
        print("ERRO: O comando 'pymol' não foi encontrado no PATH.")
    except Exception as e:
        print(f"Erro ao tentar abrir o PyMOL: {e}")

if __name__ == "__main__":
    main()
