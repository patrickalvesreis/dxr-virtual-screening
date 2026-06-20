from pymol import cmd
import csv
import sys
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES E DIRETÓRIOS
# ==============================================================================

# Usamos o diretório atual (cwd), pois você roda o PyMOL da raiz do projeto
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


def escolher_ligante_otimizado(obj_name: str, min_atoms: int = 8):
    """
    Seleciona o melhor candidato a ligante usando heurística de tamanho.

    Lógica:
      1. Seleciona apenas átomos orgânicos não-polímeros e não-solventes.
      2. Agrupa por (cadeia, resi, resn).
      3. Dos candidatos com >= min_atoms, escolhe o MENOR (assumindo Inibidor < NADPH).
      4. Se ninguém passar do filtro, escolhe o MAIOR fragmento disponível (fallback).

    Retorna:
      (selecao_pymol, (chain, resi, resn), n_atoms) ou None
    """
    # Seleção robusta: orgânicos, não-polímeros, não-solventes
    sel_all = f"{obj_name} and organic and not polymer and not solvent"
    # Se quiser testar com 'not inorganic', troque pela linha abaixo:
    # sel_all = f"{obj_name} and organic and not polymer and not solvent and not inorganic"

    try:
        model = cmd.get_model(sel_all)
    except Exception as e:
        print(f"[DEBUG] Falha em get_model para {obj_name} com seleção '{sel_all}': {e}")
        return None

    if not model.atom:
        return None

    # Agrupar átomos por (cadeia, resi, resn)
    ligandos = {}  # (chain, resi, resn) -> contagem de átomos
    for at in model.atom:
        key = (at.chain, at.resi, at.resn)
        ligandos.setdefault(key, 0)
        ligandos[key] += 1

    if not ligandos:
        return None

    # Filtrar candidatos com tamanho mínimo
    candidatos_validos = {k: v for k, v in ligandos.items() if v >= min_atoms}

    best_key = None
    best_n = None

    if candidatos_validos:
        # Cenário ideal: inibidor + NADPH, etc. Escolher o menor (inibidor tende a ser menor)
        best_key = min(candidatos_validos, key=candidatos_validos.get)
        best_n = candidatos_validos[best_key]
    else:
        # Fallback: só fragmentos pequenos; pegar o maior para não salvar lixo microscópico
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
    # Limpa o estado do PyMOL (importante em processamento em lote)
    cmd.reinitialize()

    if not MAP_CSV.exists():
        print(f"[ERRO FATAL] Arquivo CSV não encontrado em: {MAP_CSV}")
        return

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
                f"[{ligand_id}] Selecionado: Resíduo '{resn}' "
                f"(Cadeia '{chain}', ID {resi}) com {n_atoms} átomos.\n"
            )

            try:
                # Remove tudo que não é o ligante selecionado
                cmd.remove(f"not {sel}")

                # Salva o ligante isolado
                out_path = OUT_DIR / f"{ligand_id}_LIG1.pdb"
                cmd.save(str(out_path), obj_name)
                log.write(f"[{ligand_id}] SUCESSO: Salvo em {out_path}\n")
                sucessos += 1
            except Exception as e:
                log.write(f"[{ligand_id}] ERRO ao processar/salvar: {e}\n")
                erros += 1

            # Limpa o objeto da memória
            cmd.delete(obj_name)

        log.write("-" * 60 + "\n")
        log.write(f"Processamento concluído. Sucessos: {sucessos} | Erros: {erros}\n")

    # Encerra PyMOL ao final
    cmd.quit()


if __name__ == "__main__":
    processar()
