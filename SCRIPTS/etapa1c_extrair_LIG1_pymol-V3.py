from pymol import cmd
import csv
import sys
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES E DIRETÓRIOS
# ==============================================================================

# Como você roda o script da raiz do projeto, usamos o diretório atual (cwd)
# Isso evita erros onde o PyMOL não resolve o __file__ corretamente
BASE_DIR = Path.cwd()

MAP_CSV = BASE_DIR / "01_Mapeamentos" / "map_boltz2_outputs.csv"
OUT_DIR = BASE_DIR / "02_Ligantes_3D" / "pdb_from_cif_LIG1"
LOG_DIR = BASE_DIR / "02_Ligantes_3D" / "logs"
LOG_PATH = LOG_DIR / "etapa1c_extrair_LIG1_pymol.log"

# Garante que os diretórios de saída existem antes de qualquer coisa
try:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"ERRO CRÍTICO ao criar diretórios: {e}")
    sys.exit(1)


def escolher_ligante_otimizado(obj_name, min_atoms=8):
    """
    Seleciona o melhor candidato a ligante usando heurística de tamanho.
    """
   # Versão opcional, usando 'inorganic' (essa sim existe)
sel_all = f"{obj_name} and organic and not polymer and not solvent and not inorganic"

    model = cmd.get_model(sel_all)

    if not model.atom:
        return None

    # Agrupar átomos por (cadeia, resi, resn)
    ligandos = {}  # Chave: (chain, resi, resn) -> Valor: contagem de átomos

    for at in model.atom:
        key = (at.chain, at.resi, at.resn)
        ligandos.setdefault(key, 0)
        ligandos[key] += 1

    # Filtrar candidatos válidos (acima do tamanho mínimo)
    candidatos_validos = {k: v for k, v in ligandos.items() if v >= min_atoms}

    best_key = None
    best_n = None

    if candidatos_validos:
        # CORREÇÃO AQUI: Variável agora está em português consistente
        best_key = min(candidatos_validos, key=candidatos_validos.get)
        best_n = candidatos_validos[best_key]
    else:
        # Fallback: Se só tem fragmentos pequenos, pega o maior deles
        if ligandos:
            best_key = max(ligandos, key=ligandos.get)
            best_n = ligandos[best_key]

    if best_key is None:
        return None

    chain, resi, resn = best_key

    # Construir string de seleção PyMOL precisa
    if chain:
        sel = f"({obj_name} and chain {chain} and resi {resi} and resn {resn})"
    else:
        sel = f"({obj_name} and resi {resi} and resn {resn})"

    return sel, best_key, best_n


def processar():
    cmd.reinitialize()

    # Verifica se o arquivo CSV existe antes de abrir o log
    if not MAP_CSV.exists():
        print(f"[ERRO FATAL] Arquivo CSV não encontrado em: {MAP_CSV}")
        return

    # Abre o log
    with LOG_PATH.open("w", encoding="utf-8") as log:
        with MAP_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            linhas = list(reader)

        log.write(f"Iniciando extração em: {BASE_DIR}\n")
        log.write(f"Total de registros: {len(linhas)}\n")
        log.write("-" * 60 + "\n")

        sucessos = 0
        erros = 0

        for linha in linhas:
            ligand_id = linha["ligand_id"]
            cif_path_str = linha.get("cif_model_0", "")

            # Resolve o caminho do CIF relativo ao BASE_DIR se necessário
            if cif_path_str:
                cif_path = Path(cif_path_str)
                # Se o caminho no CSV for relativo, garante que resolve a partir do BASE_DIR
                if not cif_path.is_absolute():
                    cif_path = BASE_DIR / cif_path
            else:
                cif_path = None

            if not cif_path or not cif_path.exists():
                log.write(f"[{ligand_id}] ERRO: Arquivo CIF não encontrado ({cif_path})\n")
                erros += 1
                continue

            obj_name = f"complex_{ligand_id}"

            try:
                cmd.load(str(cif_path), obj_name)
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO fatal ao carregar CIF: {e}\n")
                cmd.delete(obj_name)
                erros += 1
                continue

            result = escolher_ligante_otimizado(obj_name, min_atoms=8)

            if result is None:
                log.write(f"[{ligand_id}] ERRO: Nenhum ligante orgânico detectado.\n")
                cmd.delete(obj_name)
                erros += 1
                continue

            sel, (chain, resi, resn), n_atoms = result

            log.write(
                f"[{ligand_id}] Selecionado: Resíduo '{resn}' (Cadeia '{chain}', ID {resi}) "
                f"com {n_atoms} átomos.\n"
            )

            try:
                cmd.remove(f"not {sel}")
                out_path = OUT_DIR / f"{ligand_id}_LIG1.pdb"
                cmd.save(str(out_path), obj_name)
                log.write(f"[{ligand_id}] SUCESSO: Salvo em {out_path}\n")
                sucessos += 1
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao processar/salvar: {e}\n")
                erros += 1

            cmd.delete(obj_name)

        log.write("-" * 60 + "\n")
        log.write(f"Processamento concluído. Sucessos: {sucessos} | Erros: {erros}\n")

    cmd.quit()

if __name__ == "__main__":
    processar()
