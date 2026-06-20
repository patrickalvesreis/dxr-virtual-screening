"""
SCRIPT: 06_preparar_receptor.py
AMBIENTE: REQUER 'micromamba activate docking'
DESCRIÇÃO: Usa PDB local (2JCV.pdb) no caminho especificado, limpa e prepara para Docking.
"""

import os
import sys
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import openmm.app as app

# --- CONFIGURAÇÕES ---
PDB_ID = "2JCV"

# Caminho absoluto definido pelo usuário
RAW_PDB_PATH = "/home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/docking_data/2JCV.pdb"

# Define o diretório de saída com base no local do arquivo original
OUTPUT_DIR = os.path.dirname(RAW_PDB_PATH)
OUTPUT_PDB = os.path.join(OUTPUT_DIR, f"{PDB_ID}_clean.pdb")

def main():
    print(f"--- PREPARAÇÃO DO RECEPTOR ({PDB_ID}) ---")

    # 1. Carregar PDB (Caminho Absoluto)
    if os.path.exists(RAW_PDB_PATH):
        print(f"Lendo arquivo local: {RAW_PDB_PATH}")
        fixer = PDBFixer(filename=RAW_PDB_PATH)
    else:
        print(f"[ERRO] Arquivo não encontrado no caminho específico:")
        print(f" -> {RAW_PDB_PATH}")
        print("-" * 30)
        print("Tentando baixar automaticamente da web para este local...")

        # Cria o diretório se ele não existir
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
                print(f"Diretório criado: {OUTPUT_DIR}")
            except OSError as e:
                print(f"[ERRO FATAL] Não foi possível criar o diretório: {e}")
                sys.exit(1)

        try:
            fixer = PDBFixer(pdbid=PDB_ID)
            # Nota: O PDBFixer baixa para a memória, precisamos salvar depois ou o usuário deve baixar manualmente
            print("Download bem-sucedido (na memória). Prosseguindo com a limpeza...")
        except Exception as e:
            print(f"\n[ERRO FATAL] Não foi possível baixar o PDB: {e}")
            print(f"SOLUÇÃO: Baixe manualmente 'https://files.rcsb.org/download/2JCV.pdb'")
            print(f"         e salve exatamente em: {RAW_PDB_PATH}")
            sys.exit(1)

    # 2. Identificar e Remover Cadeias/Solvente
    print("Mantendo apenas a Cadeia A (Monomero)...")
    # 2JCV tem cadeias A e B. Removemos a B.
    try:
        # Verifica se a cadeia B existe antes de tentar remover (segurança)
        chain_ids = [c.id for c in fixer.topology.chains()]
        if 'B' in chain_ids:
            fixer.removeChains(chainIds=['B'])
            print("Cadeia B removida com sucesso.")
        else:
            print("Aviso: Cadeia B não encontrada ou já removida.")
    except Exception as e:
        print(f"Aviso ao remover cadeias: {e}")

    # Encontrar falta de resíduos
    print("Buscando resíduos e átomos faltantes...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    # Adicionar o que falta
    if len(fixer.missingResidues) > 0 or len(fixer.missingAtoms) > 0:
        print(f"Reparando: {len(fixer.missingResidues)} resíduos e {len(fixer.missingAtoms)} átomos faltantes...")
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
    else:
        print("Nenhum átomo ou resíduo faltante encontrado.")

    # 3. Remover Ligantes (Limpeza total para docking cego inicial)
    print("Removendo águas e ligantes originais (Fosmidomicina/NADPH)...")
    fixer.removeHeterogens(keepWater=False)

    # 4. Salvar PDB Limpo
    print(f"Salvando receptor limpo em: {OUTPUT_PDB}")
    try:
        with open(OUTPUT_PDB, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print("-" * 30)
        print("PRONTO! Receptor preparado com sucesso.")
    except PermissionError:
        print(f"[ERRO] Sem permissão de escrita em: {OUTPUT_PDB}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar arquivo: {e}")

if __name__ == "__main__":
    main()
