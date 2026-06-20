#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adicionar_gnn_pura.py

Cria (ou atualiza) uma pasta com o conteúdo da GNN Pura
dentro de PROJETO_ORGANIZADO_FINAL, sem apagar nada do que já existe.

Arquivos esperados da GNN Pura:

- scripts/03_criar_grafos_pyg.py
- scripts/04_treinar_gnn_gpu.py
- scripts/12_gerar_predicoes_gnn.py
- outputs/grafos_dxr.pt
- outputs/modelo_gnn_best.pth (ou .pt)
- outputs/predicoes_gnn.csv
"""

import os
import shutil

# Este arquivo está em: .../Banco-de-dados-bindingDB/Descritores/scripts
BASE_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# Pasta .../Banco-de-dados-bindingDB/Descritores
BASE_DESC = os.path.dirname(BASE_SCRIPTS)

# Pasta .../Banco-de-dados-bindingDB/Descritores/outputs
BASE_OUTPUTS = os.path.join(BASE_DESC, "outputs")

# Pasta destino organizada
DESTINO_ROOT = os.path.join(BASE_DESC, "PROJETO_ORGANIZADO_FINAL")
DESTINO_GNN_PURA = os.path.join(DESTINO_ROOT, "03_GNN_Pura")


def copiar(origem_dir: str, destino_dir: str, nome_arquivo: str) -> None:
    """Copia um arquivo se ele existir."""
    os.makedirs(destino_dir, exist_ok=True)
    src = os.path.join(origem_dir, nome_arquivo)

    if os.path.exists(src):
        try:
            shutil.copy2(src, destino_dir)
            print(f"✅ Copiado: {src} -> {destino_dir}")
        except Exception as e:
            print(f"❌ Erro ao copiar {src}: {e}")
    else:
        print(f"⚠ Arquivo não encontrado (pulei): {src}")


def copiar_primeiro_que_existir(origem_dir: str, destino_dir: str, nomes_arquivos) -> None:
    """Tenta copiar o primeiro nome da lista que existir em origem_dir."""
    for nome in nomes_arquivos:
        src = os.path.join(origem_dir, nome)
        if os.path.exists(src):
            copiar(origem_dir, destino_dir, nome)
            return
    print(f"⚠ Nenhum dos arquivos {nomes_arquivos} encontrado em {origem_dir} (todos pulados).")


def main() -> None:
    print(f"--- ADICIONANDO GNN Pura EM: {DESTINO_GNN_PURA} ---")

    # Garante que a raiz organizada existe (não apaga nada)
    if not os.path.exists(DESTINO_ROOT):
        print("ℹ Pasta PROJETO_ORGANIZADO_FINAL não existe ainda; criando...")
        os.makedirs(DESTINO_ROOT, exist_ok=True)

    # Scripts da GNN Pura
    copiar(BASE_SCRIPTS, DESTINO_GNN_PURA, "03_criar_grafos_pyg.py")
    copiar(BASE_SCRIPTS, DESTINO_GNN_PURA, "04_treinar_gnn_gpu.py")
    copiar(BASE_SCRIPTS, DESTINO_GNN_PURA, "12_gerar_predicoes_gnn.py")

    # Dados / modelos da GNN Pura
    copiar(BASE_OUTPUTS, DESTINO_GNN_PURA, "grafos_dxr.pt")
    copiar_primeiro_que_existir(
        BASE_OUTPUTS,
        DESTINO_GNN_PURA,
        ["modelo_gnn_best.pth", "modelo_gnn_best.pt"],
    )
    copiar(BASE_OUTPUTS, DESTINO_GNN_PURA, "predicoes_gnn.csv")

    print("\n=== Concluído: pasta GNN Pura adicionada a PROJETO_ORGANIZADO_FINAL ===")


if __name__ == "__main__":
    main()
