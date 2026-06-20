PROJETO: Triagem Virtual DXR - Consenso IA + Física
AUTOR: Patrick
DATA: Novembro/2025
RESULTADO FINAL: R² = 0.51 (Consenso GNN + Docking)

--- GUIA DE PASTAS ---

01_Dados_Comuns/
   - Arquivos base: CSV experimental e Receptor preparado.

02_Random_Forest/
   - Benchmark inicial. Resultado ruim (Overfitting). Serve de comparação.

03-1_GNN_Pura/  <-- **MODELO PRINCIPAL DE IA**
   - Rede Neural em Grafos (PyTorch Geometric).
   - Contém o modelo treinado ('modelo_gnn_best.pth') usado no consenso.

03_GNN_Hibrida/
   - Experimento: GNN com descritores RDKit adicionais.

04_ML_COMPLETO/
   - Experimentos avançados: Descritores 3D, XGBoost e Ensembles.
   - Mostra a exaustão de tentativas puramente estatísticas.

05_Docking_Gnina/  <-- **MODELO FÍSICO**
   - Docking molecular com AutoDock Vina/Gnina acelerado por GPU.
   - Contém resultados de afinidade (CNN Affinity) corrigidos.

06_Consenso_Final/  <-- **RESULTADOS FINAIS**
   - Onde a mágica acontece: Média ponderada entre GNN Pura (03-1) e Docking (05).
   - Arquivo principal: 'TABELA_MESTRA_FINAL.csv' (Ranking dos melhores compostos).

07_Visualizacao_Misc/
   - Scripts para gerar imagens 3D no PyMOL.

--- COMO REPRODUZIR O RESULTADO ---
1. As predições de IA já estão em '03-1_GNN_Pura/predicoes_gnn.csv'.
2. Os scores de Docking já estão em '05_Docking_Gnina/resultados/resultados_docking_gnina.csv'.
3. O script '06_Consenso_Final/12_ranking_final_unificado_v2.py' cruza esses dois arquivos para gerar o ranking final.