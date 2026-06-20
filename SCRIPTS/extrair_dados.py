#!/usr/bin/env python3
import argparse, os, sys, json, csv, re
from typing import List, Dict, Iterable

TAG_RE = re.compile(r"^>\s*<([^>]+)>")

def iter_sdf_blocks(path: str) -> Iterable[List[str]]:
    """Gera blocos (lista de linhas) separados por '$$$$'."""
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "$$$$":
                yield buf
                buf = []
            else:
                buf.append(line)
        if buf:  # último bloco (caso arquivo não termine com $$$$)
            yield buf

def parse_block(block: List[str], fields: List[str]) -> Dict[str, str]:
    """
    Extrai campos de um bloco SDF.
    Suporta valores multi-linha até linha em branco ou novo tag.
    """
    want = set(fields)
    out: Dict[str, str] = {}
    i = 0
    n = len(block)

    while i < n:
        m = TAG_RE.match(block[i].strip())
        if not m:
            i += 1
            continue

        name = m.group(1)
        i += 1  # avança para a 1ª linha do valor

        # Coleta linhas de valor
        val_lines = []
        while i < n:
            s = block[i]
            # parada: linha em branco, novo tag, ou delimitador (defensivo)
            if not s.strip() or TAG_RE.match(s.strip()) or s.strip() == "$$$$":
                break
            val_lines.append(s)
            i += 1

        value = "\n".join(val_lines).strip() if val_lines else ""
        if name in want and name not in out:
            out[name] = value if value else "N/A"
        # não incrementa i aqui quando quebra por condição acima; o loop continua

    # garante todos os campos
    for f in fields:
        out.setdefault(f, "N/A")
    return out

def maybe_float_ic50(v: str):
    """Tenta converter IC50 textual para float (nM). Mantém string se não der."""
    if v in ("", "N/A"):
        return v
    s = v.replace(",", ".")
    # remove qualquer coisa que não seja parte de número/exp
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return v
    return v

def extrair_dados_sdf(caminho_arquivo: str,
                      campos: List[str]) -> List[Dict[str, object]]:
    resultados = []
    for idx, bloco in enumerate(iter_sdf_blocks(caminho_arquivo), start=1):
        data = parse_block(bloco, campos)
        row = {"Molecula_N": idx, **data}
        # tratamento especial para IC50 (nM), se estiver nos campos
        for k in list(row.keys()):
            if k.lower().startswith("ic50") and "(nM" in k:
                row["IC50_nM"] = maybe_float_ic50(str(row.pop(k)))
        resultados.append(row)
    return resultados

def print_table(rows: List[Dict[str, object]], headers: List[str]) -> None:
    if not rows:
        print("Nenhum dado foi extraído. Verifique o arquivo.")
        return
    # calcula larguras
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    # cabeçalho
    line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    # linhas
    for r in rows:
        print(" | ".join(f"{str(r.get(h, '')):<{widths[h]}}" for h in headers))

def main():
    ap = argparse.ArgumentParser(
        description="Extrai campos de um arquivo SDF (BindingDB, etc.)."
    )
    ap.add_argument("sdf", help="Caminho para o arquivo .sdf")
    ap.add_argument("-f", "--fields", nargs="*", default=["BindingDB MonomerID", "IC50 (nM)"],
                    help="Lista de campos a extrair (ex.: -f 'BindingDB MonomerID' 'IC50 (nM)')")
    ap.add_argument("--csv", action="store_true", help="Exporta CSV para stdout")
    ap.add_argument("--json", action="store_true", help="Exporta JSON para stdout")
    args = ap.parse_args()

    if not os.path.exists(args.sdf):
        print(f"Erro: arquivo não encontrado: {args.sdf}", file=sys.stderr)
        sys.exit(1)

    dados = extrair_dados_sdf(args.sdf, args.fields)

    # define ordem das colunas
    headers = ["Molecula_N"]
    # se IC50 (nM) foi mapeado para IC50_nM, coloca nome novo na ordem
    norm_fields = []
    for f in args.fields:
        if f.lower().startswith("ic50") and "(nM" in f:
            norm_fields.append("IC50_nM")
        else:
            norm_fields.append(f)
    # evita duplicatas
    for h in norm_fields:
        if h not in headers:
            headers.append(h)

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=headers)
        w.writeheader()
        for r in dados:
            w.writerow({h: r.get(h, "") for h in headers})
    elif args.json:
        json.dump([{h: r.get(h, "") for h in headers}], sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"\n--- Dados Extraídos de '{args.sdf}' ---")
        print(f"Total de moléculas: {len(dados)}\n")
        print_table(dados, headers)

if __name__ == "__main__":
    main()
