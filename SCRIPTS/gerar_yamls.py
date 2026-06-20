import os
import pandas as pd

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
csv_path = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR/Banco-de-dados-bindingDB/Descritores/PROJETO_DXR_ML_GNN/00_Dados_Originais/dataset_dxr_v2_CLEAN.csv"
output_folder = "inputs_yaml"  # pasta onde os YAMLs serão salvos

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# DADOS FIXOS (proteína + cofator)
# -----------------------------
protein_sequence = (
    "TMAHHHHHHVTNSTDGRADGRLRVVVLGSTGSIGTQALQVIADNPDRFEVVGLAAGGAHLDTLLRQRAQTGVTNIAVADEHAA"
    "QRVGDIPYHGSDAATRLVEQTEADVVLNALVGALGLRPTLAALKTGARLALANKESLVAGGSLVLRAARPGQIVPVDSEHSAL"
    "AQCLRGGTPDEVAKLVLTASGGPFRGWSAADLEHVTPEQAGAHPTWSMGPMNTLNSASLVNKGLEVIETHLLFGIPYDRIDVV"
    "VHPQSIIHSMVTFIDGSTIAQASPPDMKLPISLALGWPRRVSGAAAACDFHTASSWEFEPLDTDVFPAVELARQAGVAGGCMT"
    "AVYNAANEEAAAAFLAGRIGFPAIVGIIADVLHAADQWAVEPATVDDVLDAQRWARERAQRAVSGM"
)

nadph_smiles = "NC(=O)C1=CN(C=CC1)[CH]2O[CH](CO[P](O)(=O)O[P](O)(=O)OC[CH]3O[CH]([CH](O[P](O)(O)=O)[CH]3O)n4cnc5c(N)ncnc45)[CH](O)[CH]2O"

# -----------------------------
# LEITURA DO CSV
# -----------------------------
print(f"Lendo CSV: {csv_path}...")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # remove espaços nas extremidades

print("Colunas encontradas:", list(df.columns))

# Escolha da coluna de ID (preferência em ordem)
id_candidates = ["id", "BindingDB MonomerID", "Molecula_N"]
id_col = None
for c in id_candidates:
    if c in df.columns:
        id_col = c
        break

if id_col is None:
    print("⚠ Não encontrei nenhuma coluna de ID esperada. Vou usar o índice da linha como ID.")
else:
    print(f"Usando coluna de ID: {id_col}")

# Escolha da coluna de SMILES (preferência em ordem)
smiles_candidates = ["SMILES_CLEAN", "canonical_smiles", "isomeric_smiles"]
smiles_col = None
for c in smiles_candidates:
    if c in df.columns:
        smiles_col = c
        break

if smiles_col is None:
    raise RuntimeError("Não encontrei nenhuma coluna de SMILES esperada (SMILES_CLEAN / canonical_smiles / isomeric_smiles).")

print(f"Usando coluna de SMILES: {smiles_col}")

has_ic50 = "IC50_nM" in df.columns
if has_ic50:
    print("IC50_nM encontrado — será incluído em properties.affinity quando disponível.")
else:
    print("Coluna IC50_nM NÃO encontrada — YAMLs serão gerados sem valor numérico de afinidade.")

# -----------------------------
# LOOP DE GERAÇÃO
# -----------------------------
count = 0

for idx, row in df.iterrows():
    # -------------------------
    # 1) ID da molécula
    # -------------------------
    if id_col is not None:
        raw_id = row[id_col]
        if pd.isna(raw_id):
            mol_id = f"row_{idx}"
        else:
            # tenta converter para inteiro se for número, senão usa string direto
            try:
                mol_id = str(int(raw_id))
            except Exception:
                mol_id = str(raw_id).strip()
    else:
        mol_id = f"row_{idx}"

    # -------------------------
    # 2) SMILES da molécula (ligante B)
    # -------------------------
    smiles_b = row[smiles_col]
    if pd.isna(smiles_b) or str(smiles_b).strip() == "":
        print(f"⚠ Linha {idx}: SMILES vazio. Pulando.")
        continue

    smiles_b = str(smiles_b).strip()
    # Escapa aspas simples para não quebrar o YAML
    smiles_b_yaml = smiles_b.replace("'", "''")
    nadph_smiles_yaml = nadph_smiles.replace("'", "''")

    # -------------------------
    # 3) Afinidade (IC50 em nM, se existir)
    # -------------------------
    ic50_val = None
    if has_ic50:
        val = row["IC50_nM"]
        if not pd.isna(val):
            try:
                ic50_val = float(val)
            except Exception:
                ic50_val = None

    # Monta o bloco de afinidade
    affinity_block = "  - affinity:\n"
    affinity_block += "      target: A\n"
    affinity_block += "      binder: B\n"
    if ic50_val is not None:
        affinity_block += "      kind: IC50\n"
        affinity_block += f"      value: {ic50_val:.6g}\n"
        affinity_block += "      unit: nM\n"
    # Metadados de origem
    affinity_block += "      source:\n"
    affinity_block += '        dataset: "BindingDB_DXR_v2"\n'
    affinity_block += f'        monomer_id: "{mol_id}"\n'

    # -------------------------
    # 4) Conteúdo YAML
    # -------------------------
    yaml_content = f"""version: 1
system:
  id: "2JCV"
  description: "DXR (1-deoxy-D-xylulose 5-phosphate reductoisomerase) de Mycobacterium tuberculosis em complexo com ligante candidato e cofator NADPH."
  organism: "Mycobacterium tuberculosis"
  pdb_chains: ["A", "B"]

sequences:
  - protein:
      id: A
      name: "DXR"
      full_name: "1-deoxy-D-xylulose 5-phosphate reductoisomerase"
      pdb_id: "2JCV"
      pdb_chains: ["A", "B"]
      organism: "Mycobacterium tuberculosis"
      role: "target"
      sequence: >
        {protein_sequence}
  - ligand:
      id: B
      name: "Ligante_{mol_id}"
      role: "candidate_inhibitor"
      smiles: '{smiles_b_yaml}'
  - ligand:
      id: C
      name: "NADPH"
      pdb_ligand_id: "NDP"
      role: "cofactor"
      smiles: '{nadph_smiles_yaml}'

properties:
{affinity_block}
"""

    # -------------------------
    # 5) Salvar arquivo
    # -------------------------
    safe_filename = "".join(c for c in mol_id if c.isalnum() or c in ("-", "_")).rstrip()
    if not safe_filename:
        safe_filename = f"mol_{idx}"

    filepath = os.path.join(output_folder, f"{safe_filename}.yaml")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    count += 1

print("--------------------------------------------------")
print(f"Processo concluído. Total de arquivos gerados: {count}")
print(f"Pasta de saída: {os.path.abspath(output_folder)}")
