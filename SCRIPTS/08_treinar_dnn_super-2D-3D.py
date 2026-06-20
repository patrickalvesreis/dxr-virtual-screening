"""
SCRIPT: 08_treinar_dnn_super.py
DESCRIÇÃO: Treina uma Deep Neural Network (MLP) no dataset massivo 2D+3D.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "outputs", "super_dataset.npz")
PLOT_PATH = os.path.join(BASE_DIR, "..", "outputs", "resultado_super_dnn.png")

# Hiperparâmetros Otimizados para "High Dimension / Small Sample"
BATCH_SIZE = 32
EPOCHS = 500        # Mais épocas, pois temos muito dropout
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.5  # Crucial para evitar overfitting com tantos descritores
HIDDEN_DIM = 1024   # Camada larga para capturar combinações de bits

class SuperDNN(nn.Module):
    def __init__(self, input_dim):
        super(SuperDNN, self).__init__()

        self.net = nn.Sequential(
            # Camada 1: Compressão inicial
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM), # Estabiliza o treino
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            # Camada 2: Processamento profundo
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            # Camada 3: Refinamento
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE // 2),

            # Saída
            nn.Linear(HIDDEN_DIM // 4, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("--- TREINANDO DNN COM SUPER FEATURES (2D + 3D) ---")

    # 1. Carregar Dados
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Rode o script 07 primeiro!")

    data = np.load(DATA_PATH)
    X_np = data['X']
    y_np = data['y']

    # Normalizar Features 3D (as ECFP são 0/1, mas as 3D são contínuas)
    # Uma estratégia simples: Log1p nas features para suavizar valores extremos
    # Ou StandardScaler. Como misturamos bits e contínuos, vamos deixar a BatchNorm lidar com isso,
    # mas é bom garantir que não há NaNs.
    X_np = np.nan_to_num(X_np)

    # Converter para Tensores
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    print(f"Input Features: {X_tensor.shape[1]}")

    # 2. Split
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.85 * len(dataset)) # 85% treino para dar mais dados
    test_size = len(dataset) - train_size

    # Seed fixa para reprodutibilidade
    torch.manual_seed(42)
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Setup Modelo e GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")

    model = SuperDNN(input_dim=X_tensor.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.MSELoss()

    # Scheduler para reduzir LR se estagnar
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.5)

    # 4. Loop de Treino
    best_loss = float('inf')
    history_train = []
    history_test = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_data)
        history_train.append(train_loss)

        # Validação
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                test_loss += criterion(pred, by).item() * bx.size(0)

        test_loss /= len(test_data)
        history_test.append(test_loss)

        scheduler.step(test_loss)

        if epoch % 20 == 0:
            print(f"Ep {epoch}: Train MSE={train_loss:.4f}, Test MSE={test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            # Salvar melhor estado na memória (ou disco)
            best_model_state = model.state_dict()

    print("-" * 30)
    print(f"Melhor Test MSE: {best_loss:.4f}")

    # 5. Avaliação Final
    model.load_state_dict(best_model_state)
    model.eval()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            preds_list.extend(pred.cpu().numpy().flatten())
            targets_list.extend(by.cpu().numpy().flatten())

    r2 = r2_score(targets_list, preds_list)
    print(f"R² FINAL (SUPER DNN): {r2:.4f}")

    # 6. Gráfico
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    sns.regplot(x=targets_list, y=preds_list, color='purple')
    plt.plot([min(targets_list), max(targets_list)], [min(targets_list), max(targets_list)], 'k--')
    plt.xlabel('Real pIC50')
    plt.ylabel('Predito pIC50')
    plt.title(f'DNN Super Features (2D+3D) - R²={r2:.3f}')
    plt.savefig(PLOT_PATH)
    print(f"Gráfico salvo: {PLOT_PATH}")

if __name__ == "__main__":
    main()
