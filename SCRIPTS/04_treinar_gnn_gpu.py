"""
SCRIPT: 04_treinar_gnn_gpu.py (CORRIGIDO)
DESCRIÇÃO: Treina um modelo GCN para prever pIC50 usando GPU.
CORREÇÃO: Adicionado weights_only=False para carregar objetos PyG.
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "outputs", "grafos_dxr.pt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "modelo_gnn_best.pth")

# Hiperparâmetros
BATCH_SIZE = 16
HIDDEN_CHANNELS = 64
EPOCHS = 200
LEARNING_RATE = 0.01

# --- ARQUITETURA GCN ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        # Camadas de Convolução de Grafo
        self.conv1 = GCNConv(num_node_features, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)

        # Camada linear final (Regressão)
        self.lin = torch.nn.Linear(HIDDEN_CHANNELS, 1)

    def forward(self, x, edge_index, batch):
        # 1. Node Embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout Layer
        x = global_mean_pool(x, batch)

        # 3. Dropout
        x = F.dropout(x, p=0.5, training=self.training)

        # 4. Previsão Final
        x = self.lin(x)
        return x

def main():
    print("--- TREINAMENTO GNN (REGRESSÃO) ---")

    # 1. Configurar Dispositivo (GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de treino: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Carregar Dados (CORREÇÃO AQUI)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Rode o script 03 primeiro!")

    print(f"Carregando dataset de: {DATA_PATH}")
    # weights_only=False é necessário para carregar objetos Data() do PyG
    dataset = torch.load(DATA_PATH, weights_only=False)

    # Embaralhar e Dividir
    torch.manual_seed(42)
    # Converter para lista Python pura caso não seja
    if not isinstance(dataset, list):
        dataset = [data for data in dataset]

    np.random.shuffle(dataset)

    split = int(len(dataset) * 0.8)
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Treino: {len(train_dataset)} | Teste: {len(test_dataset)}")

    # 3. Inicializar Modelo
    num_features = dataset[0].num_node_features
    model = GCN(num_node_features=num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # 4. Loop de Treino
    best_loss = float('inf')

    print("Iniciando épocas...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y.squeeze())

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_loss = total_loss / len(train_dataset)

        # Validação
        if epoch % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch)
                    test_loss += criterion(out.squeeze(), data.y.squeeze()).item() * data.num_graphs

            avg_test_loss = test_loss / len(test_dataset)
            print(f"Epoch {epoch:03d}: Train MSE={avg_loss:.4f}, Test MSE={avg_test_loss:.4f}")

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                # Aqui weights_only não afeta salvar, mas afeta carregar
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("-" * 30)
    print(f"Melhor Test MSE: {best_loss:.4f}")

    # 5. Avaliação Final
    # Carregando pesos do modelo (seguro usar weights_only=True aqui, pois são só números)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds.extend(out.squeeze().cpu().tolist())
            targets.extend(data.y.squeeze().cpu().tolist())

    r2 = r2_score(targets, preds)
    print(f"R² Score no Teste: {r2:.4f}")
    print(f"Modelo salvo em: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
