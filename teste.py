import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ----------------------------
# 1) Modelo LSTM (many-to-one)
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        last_h = h_n[-1]          # (B, H) -> último estado da última camada
        y_hat = self.head(last_h) # (B, output_size)
        return y_hat


# ----------------------------
# 2) Dataset de janelas
# ----------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        """
        X: (N, F) features por tempo
        y: (N,) alvo por tempo (ex: close futuro ou retorno futuro)
        window: tamanho da janela
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x_win = self.X[idx : idx + self.window]          # (window, F)
        y_tgt = self.y[idx + self.window]                # alvo depois da janela
        return torch.from_numpy(x_win), torch.tensor([y_tgt])


# ----------------------------
# 3) Features + target
# ----------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém coisas simples e úteis:
    - retornos
    - médias móveis
    - volatilidade
    """
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change()
    out["logret_1"] = np.log(out["Close"]).diff()

    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # volume pode ter escala bem diferente
    out["vol_norm"] = np.log1p(out["Volume"])

    return out.dropna()


def make_target(df_feat: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Target (regressão) = retorno futuro em 'horizon' dias.
    Ex: horizon=1 => retorno de amanhã.
    """
    close = df_feat["Close"]
    y = close.shift(-horizon) / close - 1.0
    return y


# ----------------------------
# 4) Split temporal
# ----------------------------
def time_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


# ----------------------------
# 5) Treino e avaliação
# ----------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    count = 0

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabilidade
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)

    return total_loss / max(count, 1)


def predict(model, loader, device="cpu"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            y = yb.numpy().reshape(-1)
            preds.append(pred)
            trues.append(y)
    return np.concatenate(preds), np.concatenate(trues)


# ----------------------------
# MAIN
# ----------------------------
def main():
    # ======= CONFIG =======
    symbol = "BTC-USD"
    start = "2018-01-01"
    end = None  # hoje
    interval = "1d"

    window = 60        # 60 dias olhando pra trás
    horizon = 1        # prever retorno do próximo dia
    batch_size = 64
    epochs = 30
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ======= 1) Baixar dados =======
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
    df = df.dropna()
    # Garantir colunas padrão (Open, High, Low, Close, Volume)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # ======= 2) Features + target =======
    df_feat = make_features(df)
    y = make_target(df_feat, horizon=horizon)
    df_feat = df_feat.iloc[:-horizon]  # remove últimos sem target
    y = y.iloc[:-horizon]

    # selecione as features que vão para a rede
    feature_cols = ["Open", "High", "Low", "Close", "ret_1", "logret_1", "sma_10", "sma_20", "vol_10", "vol_20", "vol_norm"]
    data = df_feat[feature_cols].copy()
    data["y"] = y.values

    # ======= 3) Split temporal =======
    train_df, val_df, test_df = time_split(data, train_ratio=0.7, val_ratio=0.15)

    # ======= 4) Normalização (fit SOMENTE no treino) =======
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    y_train = train_df["y"].values
    y_val = val_df["y"].values
    y_test = test_df["y"].values

    # ======= 5) Datasets com janelas =======
    train_ds = WindowDataset(X_train, y_train, window=window)
    val_ds = WindowDataset(X_val, y_val, window=window)
    test_ds = WindowDataset(X_test, y_test, window=window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ======= 6) Modelo + loss + otimizador =======
    input_size = X_train.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ======= 7) Treino com "early stopping" simples =======
    best_val = float("inf")
    best_state = None
    patience = 7
    patience_left = patience

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        va_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(f"Epoch {epoch:02d} | train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping!")
                break

    # restaura melhor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    # ======= 8) Avaliação no teste =======
    preds, trues = predict(model, test_loader, device=device)

    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    direction_acc = np.mean((preds > 0) == (trues > 0))  # acerto do sinal do retorno

    print("\nTEST metrics")
    print("MSE:", mse)
    print("MAE:", mae)
    print("Direction accuracy:", direction_acc)

    # ======= 9) Plots =======
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss per epoch")
    plt.show()

    plt.figure()
    plt.plot(trues, label="true")
    plt.plot(preds, label="pred")
    plt.legend()
    plt.title("Pred vs True (test)")
    plt.show()


if __name__ == "__main__":
    main()
