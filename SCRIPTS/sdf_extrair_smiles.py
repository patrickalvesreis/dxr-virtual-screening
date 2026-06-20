cat > sdf_extrair_smiles.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# como rodar: micromamba run -n docking python sdf_extrair_smiles.py *(nome do arquivo).sdf -o smiles_out.csv -f "BindingDB MonomerID" "IC50 (nM)"
#pode mudar o nome do arquivo de saida de "smiles_out.csv" para *.csv





import argparse, os, sys, csv, re
from typing import List, Dict, Iterable, Tuple
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem import inchi as rd_inchi

TAG_RE = re.compile(r"^>\s*<([^>]+)>")

def iter_sdf_blocks(path: str) -> Iterable[List[str]]:
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "$$$$":
                yield buf
                buf = []
            else:
                buf.append(line)
        if buf:
            yield buf

def parse_block(block: List[str], fields: List[str]) -> Dict[str, str]:
    want = set(fields)
    out: Dict[str, str] = {}
    i, n = 0, len(block)
    while i < n:
        m = TAG_RE.match(block[i].strip())
        if not m:
            i += 1
            continue
        name = m.group(1)
        i += 1
        val_lines = []
        while i < n:
            s = block[i]
            if not s.strip() or TAG_RE.match(s.strip()) or s.strip() == "$$$$":
                break
            val_lines.append(s)
            i += 1
        value = "\n".join(val_lines).strip() if val_lines else ""
        if name in want and name not in out:
            out[name] = value if value else "N/A"
    for f in fields:
        out.setdefault(f, "N/A")
    return out

def maybe_float_ic50(v: str):
    if v in ("", "N/A"):
        return v
    s = v.replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return v
    return v

def rdkit_clean(mol, keep_all_fragments: bool, do_uncharge: bool):
    if mol is None:
        return None
    try:
        if not keep_all_fragments:
            chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()
            mol = chooser.choose(mol)
    except Exception:
        pass
    if do_uncharge:
        try:
            un = MolStandardize.rdMolStandardize.Uncharger()
            mol = un.uncharge(mol)
        except Exception:
            pass
    return mol

def rdkit_ids(mol) -> Tuple[str, str, str]:
    iso = Chem.MolToSmiles(mol, isomericSmiles=True,  canonical=True)
    can = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    try:
        ikey = rd_inchi.MolToInchiKey(mol)
    except Exception:
        ikey = ""
    return iso, can, ikey

def processar(sdf_path: str, out_csv: str, fields: List[str],
              keep_all_fragments: bool = False, no_uncharge: bool = False):
    blocks_iter = iter_sdf_blocks(sdf_path)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    base_headers = ["Molecula_N", "id", "isomeric_smiles", "canonical_smiles", "inchikey"]
    norm_fields = []
    for g in fields:
        if g.lower().startswith("ic50") and "(nM" in g:
            norm_fields.append("IC50_nM")
        else:
            norm_fields.append(g)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        headers = base_headers + [h for h in norm_fields if h not in base_headers]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()

        for idx, (mol, block) in enumerate(zip(suppl, blocks_iter), start=1):
            if mol is None:
                continue

            raw = parse_block(block, fields)
            record: Dict[str, object] = {}

            for k, v in list(raw.items()):
                if k.lower().startswith("ic50") and "(nM" in k:
                    raw.pop(k)
                    record["IC50_nM"] = maybe_float_ic50(v)
            for k in fields:
                if k.lower().startswith("ic50") and "(nM" in k:
                    continue
                record[k] = raw.get(k, "N/A")

            mol = rdkit_clean(mol, keep_all_fragments, do_uncharge=no_uncharge)

            mol_id = (mol.GetProp("BindingDB MonomerID") if mol.HasProp("BindingDB MonomerID")
                      else (mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"))

            iso, can, ik = rdkit_ids(mol)

            row = {
                "Molecula_N": idx,
                "id": mol_id,
                "isomeric_smiles": iso,
                "canonical_smiles": can,
                "inchikey": ik,
            }
            for h in norm_fields:
                if h not in row:
                    row[h] = record.get(h, "")
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(
        description="Extrai campos do SDF + gera SMILES isomérico/canônico e InChIKey (RDKit)."
    )
    ap.add_argument("sdf", help="Caminho para o arquivo .sdf")
    ap.add_argument("-o", "--out", default="smiles_out.csv", help="CSV de saída")
    ap.add_argument("-f", "--fields", nargs="*", default=["BindingDB MonomerID", "IC50 (nM)"],
                    help="Campos do SDF a incluir como colunas")
    ap.add_argument("--keep-all-fragments", action="store_true",
                    help="Não remover sais (não usar LargestFragmentChooser)")
    ap.add_argument("--no-uncharge", action="store_true",
                    help="Não tentar neutralizar (Uncharger)")
    args = ap.parse_args()

    if not os.path.exists(args.sdf):
        print(f"Erro: arquivo não encontrado: {args.sdf}", file=sys.stderr)
        sys.exit(1)

    processar(args.sdf, args.out, args.fields,
              keep_all_fragments=args.keep_all_fragments,
              no_uncharge=args.no_uncharge)

if __name__ == "__main__":
    main()
PY
chmod +x sdf_extrair_smiles.py
