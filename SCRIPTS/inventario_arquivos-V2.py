import os
import shutil

# --- CONFIGURAÇÕES ---
# Caminhos baseados na sua estrutura atual
BASE_SCRIPTS = os.path.dirname(os.path.abspath(__file__))  # .../Descritores/scripts
BASE_DESC = os.path.dirname(BASE_SCRIPTS)                  # .../Descritores
BASE_OUTPUTS = os.path.join(BASE_DESC, "outputs")          # .../Descritores/outputs
BASE_DOCKING_DATA = os.path.abspath(os.path.join(BASE_DESC, "..", "..", "docking_data"))
BASE_ROOT_OUTPUTS = os.path.abspath(os.path.join(BASE_DESC, "..", "..", "outputs"))  # outputs na raiz do projeto

# Destino
DESTINO_ROOT = os.path.join(BASE_DESC, "PROJETO_ORGANIZADO_FINAL")


def copiar(src_path, dest_folder_name, dest_file_name=None):
    if not os.path.exists(src_path):
        # Tenta achar arquivo com espaço no nome (ex: " 07_...") caso tenha sido salvo assim
        dirname, basename = os.path.split(src_path)
        alt_path = os.path.join(dirname, " " + basename)
        if os.path.exists(alt_path):
            src_path = alt_path
        else:
            print(f"⚠️  [FALTA] Origem não encontrada: {basename}")
            return

    dst_folder = os.path.join(DESTINO_ROOT, dest_folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    final_name = dest_file_name if dest_file_name else os.path.basename(src_path)
    dst_path = os.path.join(dst_folder, final_name)

    try:
        shutil.copy2(src_path, dst_path)
        print(f"✅ [COPIADO] {final_name} -> {dest_folder_name}")
    except Exception as e:
        print(f"❌ [ERRO] {final_name}: {e}")


def main():
    print(f"--- FINALIZANDO ORGANIZAÇÃO DA PASTA ---")
    print(f"Destino: {DESTINO_ROOT}\n")

    # 1. DADOS COMUNS (complementar com a estrutura do receptor limpo)
    copiar(os.path.join(BASE_DOCKING_DATA, "2JCV_clean.pdb"), "01_Dados_Comuns")

    # 2. ML COMPLETO - Ramo A (3D)
    dest_3d = "04_ML_COMPLETO/Ramo_A_3D_Geometria"
    copiar(os.path.join(BASE_SCRIPTS, "07_gerar_features_apenas_3D.py"), dest_3d)
    copiar(os.path.join(BASE_SCRIPTS, "08_treinar_dnn_3D.py"), dest_3d)
    copiar(os.path.join(BASE_OUTPUTS, "dataset_3d_only.npz"), dest_3d)
    copiar(os.path.join(BASE_OUTPUTS, "modelo_dnn_3d.pth"), dest_3d)
    copiar(os.path.join(BASE_OUTPUTS, "predicoes_teste_3d.csv"), dest_3d)
    copiar(os.path.join(BASE_OUTPUTS, "resultado_dnn_3d.png"), dest_3d)

    # 3. ML COMPLETO - Ramo B (2D Fingerprints)
    dest_2d = "04_ML_COMPLETO/Ramo_B_2D_Fingerprints"
    copiar(os.path.join(BASE_SCRIPTS, "07b_gerar_features_apenas_2D_ECFP4.py"), dest_2d)
    copiar(os.path.join(BASE_SCRIPTS, "08b_treinar_dnn_2D_ECFP4.py"), dest_2d)
    copiar(os.path.join(BASE_OUTPUTS, "dataset_2d_only.npz"), dest_2d)
    copiar(os.path.join(BASE_OUTPUTS, "predicoes_teste_2d.csv"), dest_2d)
    copiar(os.path.join(BASE_OUTPUTS, "resultado_dnn_2d.png"), dest_2d)

    # 4. ML COMPLETO - Ramo C (Ensemble)
    dest_ens = "04_ML_COMPLETO/Ramo_C_Ensemble"
    copiar(
        os.path.join(BASE_SCRIPTS, "10_ensemble_analise_(mistura2D-com-3D).py"),
        dest_ens,
        dest_file_name="10_ensemble_analise.py",
    )
    copiar(os.path.join(BASE_OUTPUTS, "resultado_ensemble_final.png"), dest_ens)

    # 5. DOCKING FINAL (resultados consolidados)
    dest_dock_res = "05_Docking_Gnina/resultados"
    copiar(os.path.join(BASE_OUTPUTS, "resultado_FINAL_docking.csv"), dest_dock_res)
    copiar(os.path.join(BASE_OUTPUTS, "correlacao_docking_experimental.png"), dest_dock_res)

    # 5b. Script de correlação refinada de docking
    copiar(os.path.join(BASE_SCRIPTS, "11_analise_correlacao_docking_v2.py"),
           "05_Docking_Gnina/scripts")

    # 6. CONSENSO FINAL (A Jóia da Coroa)
    dest_cons = "06_Consenso_Final"
    copiar(os.path.join(BASE_SCRIPTS, "12_ranking_final_unificado_v2.py"), dest_cons)

    # Tabela Mestra pode estar na raiz outputs ou em Descritores/outputs
    if os.path.exists(os.path.join(BASE_ROOT_OUTPUTS, "TABELA_MESTRA_FINAL.csv")):
        copiar(os.path.join(BASE_ROOT_OUTPUTS, "TABELA_MESTRA_FINAL.csv"), dest_cons)
    else:
        copiar(os.path.join(BASE_OUTPUTS, "TABELA_MESTRA_FINAL.csv"), dest_cons)

    # 7. VISUALIZAÇÃO / PyMOL (ver melhor ligante)
    dest_vis = "07_Visualizacao_Misc"
    copiar(os.path.join(BASE_ROOT_OUTPUTS, "ver_campeao.pml"), dest_vis)

    print("\n--- PROCESSO CONCLUÍDO ---")
    print(f"Verifique a pasta: {DESTINO_ROOT}")


if __name__ == "__main__":
    main()
