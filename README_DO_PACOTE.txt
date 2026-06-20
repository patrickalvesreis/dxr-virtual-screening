PACOTE PARA ORIENTADOR - PROJETO DXR
Data de organizacao: 2026-06-18

Objetivo
Este pacote reune os arquivos mais uteis para avaliacao do projeto de qualificacao:
dados brutos, dados curados, scripts, resultados, graficos e documentos de escrita.
Os arquivos originais do projeto nao foram modificados.

Estrutura

1. DOCUMENTOS/
Contem os documentos principais para leitura:
- qualificacao.docx
- qualificacao.pdf
- Rascunho fundamentado para a sua qualificacao sobre DXR (1).docx
- README-ME.txt do projeto organizado
- artigo BRAIN usado como referencia metodologica para organizar QSAR/IA/XAI

2. DADOS_BRUTOS/
Contem os dados de entrada essenciais:
- arquivos SDF originais dos ligantes DXR
- CSV extraido com SMILES, identificadores e IC50
- dataset_dxr_v1.csv
- dataset_dxr_v2_CLEAN.csv, versao curada com 118 moleculas unicas
- descritores_basicos.csv inicial
- estruturas do receptor 2JCV: PDB bruto, PDB limpo e receptor PDBQT

3. SCRIPTS/
Contem os scripts usados para:
- extrair dados do SDF
- calcular descritores RDKit e fingerprints ECFP
- criar grafos moleculares
- treinar Random Forest, XGBoost, DNNs e GNNs
- preparar receptor e ligantes
- rodar GNINA
- analisar docking e gerar rankings
- preparar entradas e descritores do Boltz

4. RESULTADOS/
Contem resultados, tabelas, graficos e modelos:
- PROJETO_ORGANIZADO_FINAL/ com a versao resumida e organizada
- outputs_raiz/ com ranking final, tabela mestra e grafico docking vs pIC50
- outputs_descritores/ com descritores, predicoes, datasets NPZ, modelos PTH e graficos
- gnina_results_sdf/ com os SDFs gerados pelo docking GNINA
- ligands_pdbqt/ com ligantes preparados para docking
- feature_engineering_data/ com descritores RDKit e ECFP4 do subprojeto curado
- ml_classico_baselines/ com metricas, modelos e graficos XGBoost
- boltz_resumo_descritores/ com mapeamento, logs e descritores hibridos extraidos do Boltz

Observacoes importantes

1. O projeto completo Banco-de-dados-bindingDB possui cerca de 3.5 GB.
   Para evitar um pacote excessivamente grande, os resultados completos e pesados do Boltz
   nao foram copiados integralmente. Foram incluidos o mapeamento, logs, scripts e descritores
   hibridos ja extraidos, que sao os arquivos mais uteis para avaliacao e reproducao parcial.

2. O dataset mais recomendado para a qualificacao e:
   DADOS_BRUTOS/dataset_dxr_v2_CLEAN.csv
   Ele contem 118 moleculas unicas apos limpeza, remocao de sais e remocao de duplicatas.

3. Os resultados principais para discutir na qualificacao estao em:
   RESULTADOS/ml_classico_baselines/
   RESULTADOS/outputs_descritores/
   RESULTADOS/outputs_raiz/
   RESULTADOS/PROJETO_ORGANIZADO_FINAL/

4. Alguns scripts possuem caminhos absolutos antigos do computador original.
   Para reexecutar, pode ser necessario ajustar variaveis como PROJECT_DIR, INPUT_FILE,
   OUTPUT_DIR e caminhos de ambientes micromamba/pixi.

Resumo dos principais resultados preliminares
- XGBoost classificacao: ROC-AUC aproximadamente 0.842 e F1 aproximadamente 0.800.
- XGBoost regressao: R2 aproximadamente 0.354, RMSE aproximadamente 0.680.
- Random Forest ECFP4: baixo desempenho no teste, R2 aproximadamente 0.053.
- DNN 2D: R2 aproximadamente 0.249.
- DNN 3D: R2 aproximadamente 0.074.
- Ensemble DNN 2D+3D: R2 aproximadamente 0.309.
- GNN pura/GCN: Pearson aproximadamente 0.731 e R2 linear aproximadamente 0.535.
- Docking GNINA isolado: baixa correlacao com pIC50, mas util como filtro estrutural.
- Consenso GNN + docking: Pearson aproximadamente 0.670 e R2 linear aproximadamente 0.449.

