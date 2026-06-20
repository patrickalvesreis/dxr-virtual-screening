
# --- SCRIPT AUTOMÁTICO PARA PYMOL ---
reinitialize

# A. Carregar Estruturas
load /home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/docking_data/receptor.pdbqt, receptor
load /home/patrick/1_PROJETO_DOCAGEM/Projeto_DXR_1-deoxy-D-xylulose-5-phosphate-reductoisomerase/docking_data/gnina_results/gnina_50181153.sdf, ligante

# B. Estilo da Proteína
bg_color white
hide everything
show cartoon, receptor
color white, receptor
set cartoon_transparency, 0.3

# C. Estilo do Ligante
show sticks, ligante
color green, ligante
util.cbag ligante

# D. Definir o "Bolso" (Pocket)
select pocket, (receptor within 5 of ligante)
show sticks, pocket
color cyan, pocket
util.cbac pocket
deselect

# E. Interações Polares
dist interacoes, ligante, pocket, mode=2
color magenta, interacoes
set dash_gap, 0.3
set dash_width, 3.0
set dash_radius, 0.1

# F. Labels
label n. CA and pocket, "%s-%s" % (resn, resi)
set label_color, black
set label_size, 14
set label_position, (0, 0, 10)

# G. Zoom final
zoom ligante, 8
    