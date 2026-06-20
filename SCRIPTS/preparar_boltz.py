# [DIR: ~/docking_boltz]
import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1
import os
import yaml

# --- Configurações ---
PROTEIN_PDB = "2JCV_clean.pdb"
LIGANDS_CSV = "dataset_dxr_v2_CLEAN.csv"
OUTPUT_DIR = "inputs_yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_protein_sequence(pdb_file):
    """Extrai a sequência de aminoácidos do PDB."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Assumindo que a primeira cadeia é a proteína principal
    # Se houver mais cadeias, precisamos concatenar ou tratar separadamente
    first_model = structure[0]
    sequences = []

    for chain in first_model:
        # Pega apenas resíduos padrão (aminoácidos)
        residues = [res for res in chain if PDB.is_aa(res)]
        seq = "".join([seq1(res.get_resname()) for res in residues])
        print(f"Cadeia {chain.id}: {len(seq)} resíduos detectados.")
        sequences.append(seq)

    # Retorna a primeira sequência válida (ou a mais longa)
    return max(sequences, key=len) if sequences else None

def gerar_inputs():
    print("--- 1. Extraindo sequência da proteína ---")
    prot_seq = get_protein_sequence(PROTEIN_PDB)
    if not prot_seq:
        print("ERRO: Não foi possível extrair a sequência do PDB.")
        return
    print(f"Sequência da proteína (início): {prot_seq[:20]}...")

    print("\n--- 2. Lendo ligantes ---")
    df = pd.read_csv(LIGANDS_CSV)

    # Identificar coluna de SMILES (usando a limpa que criamos)
    col_smiles = 'SMILES_CLEAN' if 'SMILES_CLEAN' in df.columns else 'SMILES'
    col_id = 'ID' if 'ID' in df.columns else 'id'

    print(f"Gerando inputs para {len(df)} ligantes...")

    batch_list = []

    for idx, row in df.iterrows():
        lig_id = str(row.get(col_id, f"lig_{idx}"))
        smiles = row[col_smiles]

        # Estrutura do YAML para o Boltz (formato comum)
        # Ajustado para Boltz 2 (verifique a documentação se der erro de schema)
        entry = {
            "id": f"complexo_{lig_id}",
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": prot_seq
                    }
                }
            ],
            "ligands": [
                {
                    "id": "L",
                    "smiles": smiles
                }
            ]
        }

        # Salvar arquivo individual
        filename = os.path.join(OUTPUT_DIR, f"{lig_id}.yaml")
        with open(filename, "w") as f:
            yaml.dump(entry, f, sort_keys=False)

        batch_list.append(filename)

    print(f"--- Concluído! {len(batch_list)} arquivos .yaml criados em '{OUTPUT_DIR}' ---")
    print("Exemplo de comando para rodar um deles:")
    print(f"boltz predict {batch_list[0]} --out_dir predictions")

if __name__ == "__main__":
    gerar_inputs()
