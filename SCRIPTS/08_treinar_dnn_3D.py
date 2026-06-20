"""
SCRIPT: 08_treinar_dnn_3D.py
AMBIENTE: 'micromamba activate chem_gnn'
DESCRIÇÃO: Treina uma DNN focada nos descritores 3D (819 features).
           Salva o modelo e as previsões para uso em Ensemble.
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
DATASET_PATH = os.path.join(BASE_DIR, "..", "outputs", "dataset_3d_only.npz")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "modelo_dnn_3d.pth")
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "resultado_dnn_3d.png")
PREDS_SAVE_PATH = os.path.join(BASE_DIR, "..", "outputs", "predicoes_teste_3d.csv")

# Hiperparâmetros ajustados para dataset pequeno
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 400
DROPOUT_RATE = 0.4  # Alto para evitar overfitting
WEIGHT_DECAY = 1e-4 # Regularização L2

# --- ARQUITETURA DA REDE ---
class Net3D(nn.Module):
    def __init__(self, input_dim):
        super(Net3D, self).__init__()

        # Camada 1: Compressão inicial
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) # Estabiliza o treino

        # Camada 2: Refinamento
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # Camada 3: Saída
        self.out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x

def main():
    print("--- TREINANDO DNN: ESPECIALISTA 3D ---")

    # 1. Carregar Dados
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Rode o script 07 primeiro! {DATASET_PATH} não existe.")

    data = np.load(DATASET_PATH)
    X = data['X']
    y = data['y']
    ids = data['ids'] # Importante para rastrear quem é quem

    input_dim = X.shape[1]
    print(f"Features de entrada: {input_dim}")

    # 2. Split Treino/Teste (80/20)
    # Usamos random_state fixo para garantir que o Test set seja O MESMO do modelo 2D (importante pro Ensemble)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )

    # Converter para tensores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    # Dataloader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Inicializar Modelo
    model = Net3D(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # 4. Loop de Treino
    loss_history = []
    test_loss_history = []

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

        # Avaliação
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()

        loss_history.append(epoch_loss / len(train_loader))
        test_loss_history.append(test_loss)

        if epoch % 20 == 0:
            print(f"Ep {epoch}: Train MSE={loss_history[-1]:.4f}, Test MSE={test_loss:.4f}")

    # 5. Avaliação Final
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    r2 = r2_score(y_true, final_preds)
    mse = mean_squared_error(y_true, final_preds)

    print("-" * 30)
    print(f"R² FINAL (3D ONLY): {r2:.4f}")
    print(f"MSE FINAL: {mse:.4f}")

    # 6. Salvar Resultados para Ensemble
    df_preds = pd.DataFrame({
        'ID': id_test,
        'Real_pIC50': y_true.flatten(),
        'Pred_3D': final_preds.flatten()
    })
    df_preds.to_csv(PREDS_SAVE_PATH, index=False)
    print(f"Previsões salvas em: {PREDS_SAVE_PATH}")

    # Salvar Modelo
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, final_preds, alpha=0.6, color='purple')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel("Real pIC50")
    plt.ylabel("Predito 3D")
    plt.title(f"Modelo 3D (R²={r2:.3f})")
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Gráfico salvo: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()
