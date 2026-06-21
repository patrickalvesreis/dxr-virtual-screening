# DXR Virtual Screening

Pipeline computacional para priorização de potenciais inibidores da **1-deoxy-D-xylulose-5-phosphate reductoisomerase (DXR)** usando curadoria molecular, descritores RDKit, fingerprints ECFP4, modelos clássicos de machine learning, redes neurais profundas, Graph Neural Networks (GNNs), docking molecular com GNINA e consenso de ranqueamento.

> **Status:** projeto de pesquisa em estágio experimental.  
> Os resultados aqui apresentados são preliminares e devem ser interpretados como priorização computacional, não como validação experimental de atividade biológica.

---

## Objetivo

Este repositório reúne dados, scripts e resultados de um fluxo de *virtual screening* aplicado à enzima DXR. O objetivo é combinar diferentes fontes de evidência computacional para priorizar moléculas candidatas:

- atividade experimental reportada, convertida para pIC50;
- descritores moleculares 2D calculados com RDKit;
- fingerprints ECFP4;
- modelos QSAR clássicos, como Random Forest e XGBoost;
- modelos baseados em redes neurais profundas;
- Graph Neural Networks para representação molecular;
- docking molecular com GNINA;
- consenso final entre predição baseada em ligante e evidência estrutural.

---

## Contexto biológico

A DXR participa da via não-mevalonato/MEP de biossíntese de isoprenoides. Essa via está presente em diversos microrganismos e ausente em humanos, o que torna enzimas como a DXR alvos interessantes para descoberta de novos compostos antimicrobianos.

Este projeto usa a estrutura **2JCV** como receptor de referência para as etapas estruturais e docking molecular.

---

## Estrutura do repositório

```text
.
├── DADOS_BRUTOS/
│   ├── 2JCV.pdb
│   ├── 2JCV_clean.pdb
│   ├── receptor.pdbqt
│   ├── Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase.sdf
│   ├── Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase_2.sdf
│   ├── dataset_dxr_v1.csv
│   ├── dataset_dxr_v2_CLEAN.csv
│   ├── descritores_basicos.csv
│   └── smiles_out_Ligantes_1-deoxy-D-xylulose-5-phosphate-reductoisomerase.csv
│
├── SCRIPTS/
│   ├── 00_limpar_dataset_professor.py
│   ├── 01_calculo_descritores.py
│   ├── 02_qsar_xgboost.py
│   ├── 03_criar_grafos_pyg.py
│   ├── 04_treinar_gnn_gpu.py
│   ├── 07_preparar_ligantes.py
│   ├── 09_rodar_gnina_docking.py
│   ├── 12_ranking_final_unificado.py
│   ├── 13_consenso_final.py
│   └── outros scripts auxiliares
│
├── RESULTADOS/
│   ├── PROJETO_ORGANIZADO_FINAL/
│   ├── boltz_resumo_descritores/
│   ├── feature_engineering_data/
│   ├── gnina_results_sdf/
│   ├── ligands_pdbqt/
│   ├── ml_classico_baselines/
│   ├── outputs_descritores/
│   └── outputs_raiz/
│
├── INVENTARIO_ARQUIVOS.txt
└── README_DO_PACOTE.txt
