"""
SCRIPT: 04_treinar_gnn_hibrida.py
DESCRIÇÃO: Treina modelo Híbrido (GAT + RDKit Descriptors) na GPU.
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from sklearn.metrics import r2_score
import numpy as np

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "outputs", "grafos_hibridos_dxr.pt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "modelo_hibrido_best.pth")

# Hiperparâmetros (Ajustados para modelo mais complexo)
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128    # Mais neurônios
HEADS = 4                # Cabeças de atenção (GAT)
DROPOUT = 0.4            # Dropout alto para evitar overfitting
EPOCHS = 300
LR = 0.001
WEIGHT_DECAY = 1e-4      # Regularização L2

class HybridGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_global_features):
        super().__init__()

        # --- Ramo do Grafo (GAT) ---
        # GATv2Conv é mais poderoso que GCN
        self.conv1 = GATv2Conv(num_node_features, HIDDEN_CHANNELS, heads=HEADS, dropout=DROPOUT)
        # A saída será HIDDEN * HEADS, então precisamos projetar
        self.conv2 = GATv2Conv(HIDDEN_CHANNELS * HEADS, HIDDEN_CHANNELS, heads=1, dropout=DROPOUT)

        # --- Ramo Global (MLP para Descritores) ---
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_global_features, HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT)
        )

        # --- Fusão ---
        # Grafo (Hidden) + Global (Hidden)
        self.fusion_lin = torch.nn.Linear(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS)
        self.out_lin = torch.nn.Linear(HIDDEN_CHANNELS, 1)

    def forward(self, x, edge_index, batch, global_x):
        # 1. Processar Grafo
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU geralmente funciona melhor que ReLU em GAT
        x = F.dropout(x, p=DROPOUT, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Pooling (Juntar nós em uma representação de grafo)
        # Usar Mean e Max juntos as vezes ajuda, mas vamos de Mean padrão
        x_graph = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 2. Processar Globais
        x_global = self.global_mlp(global_x)  # [batch_size, hidden_channels]

        # 3. Concatenar
        x_combined = torch.cat([x_graph, x_global], dim=1) # [batch, 2*hidden]

        # 4. Decisão Final
        x_final = self.fusion_lin(x_combined)
        x_final = F.relu(x_final)
        x_final = F.dropout(x_final, p=DROPOUT, training=self.training)

        out = self.out_lin(x_final)
        return out

def main():
    print("--- TREINO HÍBRIDO (GAT + RDKIT) NA GPU ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Carregar (com weights_only=False)
    if not os.path.exists(DATA_PATH): raise FileNotFoundError("Rode script 03_hibrido primeiro")
    dataset = torch.load(DATA_PATH, weights_only=False)

    # Info de dimensão
    num_node_features = dataset[0].x.shape[1]
    num_global_features = dataset[0].global_x.shape[1]
    print(f"Features: No={num_node_features}, Global={num_global_features}")

    # Split
    torch.manual_seed(42)
    if not isinstance(dataset, list): dataset = [d for d in dataset]
    np.random.shuffle(dataset)
    split = int(len(dataset) * 0.8)
    train_loader = DataLoader(dataset[:split], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset[split:], batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    model = HybridGNN(num_node_features, num_global_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # AdamW é melhor
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    best_loss = float('inf')

    print("Treinando...")
    for epoch in range(EPOCHS):
        model.train()
        loss_epoch = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # Passar global_x explicitamente
            out = model(data.x, data.edge_index, data.batch, data.global_x)
            loss = criterion(out.squeeze(), data.y.squeeze())
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * data.num_graphs

        avg_train_loss = loss_epoch / len(dataset[:split])

        # Eval
        if epoch % 10 == 0:
            model.eval()
            loss_test = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch, data.global_x)
                    loss_test += criterion(out.squeeze(), data.y.squeeze()).item() * data.num_graphs
            avg_test_loss = loss_test / len(dataset[split:])

            # Atualizar LR
            scheduler.step(avg_test_loss)
            curr_lr = optimizer.param_groups[0]['lr']

            print(f"Ep {epoch:03d}: Train={avg_train_loss:.4f}, Test={avg_test_loss:.4f} | LR={curr_lr:.2e}")

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("-" * 30)
    print(f"Melhor Test MSE: {best_loss:.4f} (RMSE: {np.sqrt(best_loss):.4f})")

    # Final Score
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data.global_x)
            preds.extend(out.squeeze().cpu().tolist())
            targets.extend(data.y.squeeze().cpu().tolist())

    print(f"R² Final: {r2_score(targets, preds):.4f}")

if __name__ == "__main__":
    main()
