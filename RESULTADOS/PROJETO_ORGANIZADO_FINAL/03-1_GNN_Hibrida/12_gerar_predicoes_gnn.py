"""
SCRIPT: 12_gerar_predicoes_gnn.py (CORRIGIDO)
DESCRIÇÃO: Carrega o modelo GNN treinado e salva predições em CSV.
"""
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos para o modelo SIMPLES (GCN)
# Se você quiser usar o modelo HÍBRIDO, precisaria adaptar este script para a classe HybridGNN
DATA_PATH = os.path.join(BASE_DIR, "..", "outputs", "grafos_dxr.pt")
MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "modelo_gnn_best.pth")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "outputs", "predicoes_gnn.csv")

HIDDEN_CHANNELS = 64

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.lin = torch.nn.Linear(HIDDEN_CHANNELS, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def main():
    print("--- GERANDO PREDIÇÕES GNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(DATA_PATH):
        print(f"[ERRO] Dataset não encontrado: {DATA_PATH}")
        return

    # Carregar dados
    # weights_only=False é necessário para carregar objetos Data() do PyG
    dataset = torch.load(DATA_PATH, weights_only=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Carregar modelo
    # CORREÇÃO: Usar a variável correta para o número de features
    num_features = dataset[0].num_node_features
    model = GCN(num_features).to(device)

    if os.path.exists(MODEL_PATH):
        print(f"Carregando modelo de: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print(f"[ERRO] Modelo não encontrado em: {MODEL_PATH}")
        return

    model.eval()

    results = []

    print("Calculando predições...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.item()

            # Extração robusta do ID
            if hasattr(data, 'mol_id'):
                if isinstance(data.mol_id, torch.Tensor):
                    mol_id = str(data.mol_id.item())
                elif isinstance(data.mol_id, list):
                    mol_id = str(data.mol_id[0])
                else:
                    mol_id = str(data.mol_id)
            else:
                mol_id = "Unknown"

            real = data.y.item()

            results.append({"ID": mol_id, "pIC50_Real": real, "pIC50_GNN": pred})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Sucesso! Predições salvas em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
