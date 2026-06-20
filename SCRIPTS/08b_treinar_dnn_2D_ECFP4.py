"""
SCRIPT: 08b_treinar_dnn_2D_ECFP4.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Treina DNN com ECFP4.
           Salva predicoes_teste_2d.csv para comparar com o 3D.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURAÇÕES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "outputs", "dataset_2d_only.npz")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "modelo_dnn_2d.pth")
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "resultado_dnn_2d.png")
PREDS_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "predicoes_teste_2d.csv")

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 300
DROPOUT_RATE = 0.3 # Um pouco menor pois 2D é mais estável

# --- ARQUITETURA ---
class Net2D(nn.Module):
    def __init__(self, input_dim):
        super(Net2D, self).__init__()
        # ECFP é esparso, primeira camada pode ser larga
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 32)

        self.out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x

def main():
    print("--- TREINANDO DNN: ESPECIALISTA 2D (ECFP4) ---")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Rode o script 07b primeiro!")

    data = np.load(DATASET_PATH)
    X = data['X']
    y = data['y']
    ids = data['ids']

    input_dim = X.shape[1]

    # Split IDÊNTICO ao do 3D (random_state=42)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net2D(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(train_loader))

        if epoch % 50 == 0:
            # Avaliação rápida
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t)
                test_mse = criterion(test_pred, y_test_t).item()
            print(f"Ep {epoch}: Train MSE={loss_history[-1]:.4f}, Test MSE={test_mse:.4f}")

    # Avaliação Final
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    r2 = r2_score(y_true, final_preds)
    mse = mean_squared_error(y_true, final_preds)

    print("-" * 30)
    print(f"R² FINAL (2D ONLY): {r2:.4f}")

    # Salvar Previsões
    df_preds = pd.DataFrame({
        'ID': id_test,
        'Real_pIC50': y_true.flatten(),
        'Pred_2D': final_preds.flatten()
    })
    df_preds.to_csv(PREDS_SAVE_PATH, index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, final_preds, alpha=0.6, color='green')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel("Real pIC50")
    plt.ylabel("Predito 2D")
    plt.title(f"Modelo 2D - ECFP4 (R²={r2:.3f})")
    plt.savefig(PLOT_SAVE_PATH)

if __name__ == "__main__":
    main()
