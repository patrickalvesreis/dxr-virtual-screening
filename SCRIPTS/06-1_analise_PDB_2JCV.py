import os
import numpy as np

# Caminho do PDB original (antes da limpeza)
PDB_FILE = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/docking_data/2JCV.pdb"

def find_ligand_center(pdb_path, res_name):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("HETATM") and res_name in line:
                # Colunas PDB: x=30-38, y=38-46, z=46-54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    pass

    if not coords:
        return None

    center = np.mean(coords, axis=0)
    return center

if os.path.exists(PDB_FILE):
    print(f"Lendo {PDB_FILE}...")

    # Tenta achar Fosmidomicina (FOM)
    center_fom = find_ligand_center(PDB_FILE, "FOM")
    if center_fom is not None:
        print(f"\n✅ LIGANTE ENCONTRADO (FOM):")
        print(f"   CENTER_X = {center_fom[0]:.2f}")
        print(f"   CENTER_Y = {center_fom[1]:.2f}")
        print(f"   CENTER_Z = {center_fom[2]:.2f}")
    else:
        print("\n❌ FOM não encontrado.")

    # Tenta achar NADPH (NDP) como backup
    center_ndp = find_ligand_center(PDB_FILE, "NDP")
    if center_ndp is not None:
        print(f"\n⚠️ COFATOR ENCONTRADO (NDP):")
        print(f"   CENTER_X = {center_ndp[0]:.2f}")
        print(f"   CENTER_Y = {center_ndp[1]:.2f}")
        print(f"   CENTER_Z = {center_ndp[2]:.2f}")
else:
    print("Arquivo PDB não encontrado.")
