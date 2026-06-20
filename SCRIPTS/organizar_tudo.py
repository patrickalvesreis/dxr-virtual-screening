#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
organizar_tudo.py

Organiza os arquivos principais do projeto DXR em uma cópia limpa
dentro da pasta `PROJETO_ORGANIZADO_FINAL`, sem mexer na estrutura original.

COMO USAR (no seu ambiente):

1. Abrir o ambiente Pixi:
   cd ~/chem_gnn_pixi
   pixi shell

2. Ir para a pasta de scripts de descritores:
   cd /home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/Banco-de-dados-bindingDB/Descritores/scripts

3. Rodar:
   python organizar_tudo.py

⚠ O script APAGA a pasta `PROJETO_ORGANIZADO_FINAL` anterior (se existir)
e recria tudo do zero com a estrutura organizada.
"""

import os
import shutil

# --- CONFIGURAÇÕES DE CAMINHOS ---

# Este arquivo está em: .../Banco-de-dados-bindingDB/Descritores/scripts
BASE_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# Pasta .../Banco-de-dados-bindingDB/Descritores
BASE_DESC = os.path.dirname(BASE_SCRIPTS)

# Pasta de outputs ligada aos scripts de descritores:
# .../Banco-de-dados-bindingDB/Descritores/outputs
BASE_OUTPUTS = os.path.join(BASE_DESC, "outputs")

# Raiz do projeto DXR:
# .../Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DESC, "..", ".."))

# Outputs "globais" do projeto (onde ficam, por exemplo, resultados_docking_gnina.csv):
# .../Projeto_DXR_.../outputs
PROJECT_OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")

# Pasta de dados de docking:
# .../Projeto_DXR_.../docking_data
BASE_DOCKING = os.path.join(PROJECT_ROOT, "docking_data")

# Pasta destino organizada (será criada / recriada)
DESTINO_ROOT = os.path.join(BASE_DESC, "PROJETO_ORGANIZADO_FINAL")


def copiar(origem_dir: str, destino_rel: str, nome_arquivo: str) -> None:
    """
    Copia um arquivo de `origem_dir` para `DESTINO_ROOT/destino_rel`, se existir.
    """
    dst_folder = os.path.join(DESTINO_ROOT, destino_rel)
    os.makedirs(dst_folder, exist_ok=True)

    src = os.path.join(origem_dir, nome_arquivo)
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst_folder)
            print(f"✅ Copiado: {nome_arquivo} -> {destino_rel}/")
        except Exception as e:
            print(f"❌ Erro ao copiar {nome_arquivo}: {e}")
    else:
        print(f"⚠ Arquivo não encontrado (pulei): {src}")


def copiar_primeiro_que_existir(origem_dir: str, destino_rel: str, nomes_arquivos) -> None:
    """
    Tenta copiar o primeiro arquivo da lista `nomes_arquivos` que existir em `origem_dir`.
    Útil para lidar com versões antigas/novas de nomes de script.
    """
    for nome in nomes_arquivos:
        src = os.path.join(origem_dir, nome)
        if os.path.exists(src):
            copiar(origem_dir, destino_rel, nome)
            return
    print(f"⚠ Nenhum dos arquivos {nomes_arquivos} encontrado em {origem_dir} (todos pulados).")


def main() -> None:
    print(f"--- ORGANIZANDO PROJETO EM: {DESTINO_ROOT} ---")

    # Limpar versão anterior para evitar mistura de arquivos antigos/novos
    if os.path.exists(DESTINO_ROOT):
        print("ℹ Limpando versão anterior de PROJETO_ORGANIZADO_FINAL...")
        shutil.rmtree(DESTINO_ROOT)

    # 1. DADOS COMUNS
    print("\n--- 1. Dados Comuns ---")
    copiar(BASE_OUTPUTS, "01_Dados_Comuns", "descritores_basicos.csv")

    # 2. RANDOM FOREST (ECFP / RF clássico)
    print("\n--- 2. Random Forest ---")
    copiar(BASE_SCRIPTS, "02_Random_Forest", "05_comparar_rf_scikit.py")
    copiar(BASE_OUTPUTS, "02_Random_Forest", "ecfp_deepchem.csv")
    copiar(BASE_OUTPUTS, "02_Random_Forest", "comparacao_rf_seaborn.png")

    # 3. GNN HÍBRIDA (Grafos + descritores)
    print("\n--- 3. GNN Híbrida ---")
    copiar(BASE_SCRIPTS, "03_GNN_Hibrida", "03_criar_grafos_hibridos.py")
    copiar(BASE_SCRIPTS, "03_GNN_Hibrida", "04_treinar_gnn_hibrida.py")
    copiar(BASE_SCRIPTS, "03_GNN_Hibrida", "12_gerar_predicoes_gnn.py")
    copiar(BASE_OUTPUTS, "03_GNN_Hibrida", "grafos_hibridos_dxr.pt")
    copiar(BASE_OUTPUTS, "03_GNN_Hibrida", "modelo_gnn_best.pth")
    copiar(BASE_OUTPUTS, "03_GNN_Hibrida", "predicoes_gnn.csv")

    # 4. DNN / XGBoost (Super Features 2D + 3D)
    print("\n--- 4. DNN e XGBoost ---")
    # Script de geração das super-features: versão nova (-2D-3D) e fallback para nome antigo
    copiar_primeiro_que_existir(
        BASE_SCRIPTS,
        "04_DNN_XGBoost",
        ["07_gerar_super_features-2D-3D.py", "07_gerar_super_features.py"],
    )
    # Script principal de XGBoost
    copiar(BASE_SCRIPTS, "04_DNN_XGBoost", "09_xgboost-2D-3D.py")
    # Script extra de tentativa (pode não existir mais, mas não quebra)
    copiar(BASE_SCRIPTS, "04_DNN_XGBoost", "09_salvacao_xgboost.py")
    # Saídas importantes
    copiar(BASE_OUTPUTS, "04_DNN_XGBoost", "dataset_super_features.csv")
    copiar(BASE_OUTPUTS, "04_DNN_XGBoost", "super_dataset.npz")
    copiar(BASE_OUTPUTS, "04_DNN_XGBoost", "resultado_xgboost.png")

    # 5. DOCKING GNINA
    print("\n--- 5. Docking Gnina ---")
    # Scripts (aceita 06_preparar_receptor.py ou 06_preparar_receptor-2JCV.py)
    copiar_primeiro_que_existir(
        BASE_SCRIPTS,
        "05_Docking_Gnina/scripts",
        ["06_preparar_receptor.py", "06_preparar_receptor-2JCV.py"],
    )
    copiar(BASE_SCRIPTS, "05_Docking_Gnina/scripts", "07_preparar_ligantes.py")
    copiar(BASE_SCRIPTS, "05_Docking_Gnina/scripts", "09_rodar_gnina_docking_final.py")
    copiar(BASE_SCRIPTS, "05_Docking_Gnina/scripts", "11_analise_final_gnina.py")

    # Dados do receptor
    copiar(BASE_DOCKING, "05_Docking_Gnina/dados", "receptor.pdbqt")

    # Resultados de docking:
    # resultados_docking_gnina.csv ficou na pasta outputs do projeto, NÃO em Descritores/outputs
    if os.path.exists(os.path.join(PROJECT_OUTPUTS, "resultados_docking_gnina.csv")):
        copiar(PROJECT_OUTPUTS, "05_Docking_Gnina/resultados", "resultados_docking_gnina.csv")
    else:
        # fallback (caso você tenha uma cópia antiga em Descritores/outputs)
        copiar(BASE_OUTPUTS, "05_Docking_Gnina/resultados", "resultados_docking_gnina.csv")

    copiar(PROJECT_OUTPUTS, "05_Docking_Gnina/resultados", "resultado_FINAL_docking.csv")
    copiar(PROJECT_OUTPUTS, "05_Docking_Gnina/resultados", "correlacao_docking_experimental.png")

    # Criar pastas "vazias" para indicar onde ficam SDFs e ligantes,
    # sem precisar copiar tudo (para não ficar gigante).
    os.makedirs(os.path.join(DESTINO_ROOT, "05_Docking_Gnina/dados/gnina_results_SDFs"), exist_ok=True)
    print("ℹ Nota: As pastas de SDFs (gnina_results) e ligantes_pdbqt não foram copiadas para economizar espaço.")

    # 6. CONSENSO FINAL
    print("\n--- 6. Consenso Final ---")
    copiar(BASE_SCRIPTS, "06_Consenso_Final", "13_consenso_final.py")
    copiar(BASE_OUTPUTS, "06_Consenso_Final", "modelo_consenso.png")

    print("\n" + "=" * 40)
    print(f"CONCLUSÃO: Arquivos organizados em:\n{DESTINO_ROOT}")
    print("=" * 40)


if __name__ == "__main__":
    main()
