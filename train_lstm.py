"""
Pipeline completo para previsão de preços com LSTM utilizando PyTorch e dados do Yahoo Finance.

Etapas implementadas:
1. Coleta e pré-processamento dos dados (biblioteca yfinance + normalização).
2. Construção, treinamento e avaliação do modelo com métricas MAE, RMSE e MAPE.
3. Salvamento do modelo treinado para posterior inferência.
"""

import time
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yfinance as yf


class StockPriceDataset(Dataset):
    """Dataset que contém janelas temporais (inputs) e o valor futuro (targets)."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    """Modelo LSTM simples mapeando vetores de features para o próximo valor."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.regressor(last_hidden)


@dataclass
class TrainConfig:
    """Agrupa todos os hiperparâmetros e parâmetros de dados do experimento."""

    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    feature: str = "Close"
    sequence_length: int = 60
    train_ratio: float = 0.85
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "stock_lstm_checkpoint.pt"


def download_price_series(symbol: str, start_date: str, end_date: str, feature: str) -> torch.Tensor:
    """
    Faz o download dos dados históricos via yfinance e devolve apenas a coluna desejada.

    Returns:
        Tensor 1-D com os valores cronológicos do feature escolhido.
    """
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' não encontrada no dataset retornado.")
    values = torch.tensor(df[feature].values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Dataset retornou vazio. Verifique símbolo e datas.")
    return values


def normalize_with_stats(series: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Aplicação direta de normalização z-score utilizando média e desvio informados."""
    safe_std = std if std > 1e-12 else 1.0
    return (series - mean) / safe_std


def normalize_series(series: torch.Tensor, train_ratio: float) -> Tuple[torch.Tensor, float, float]:
    """
    Calcula média/desvio apenas com a porção de treino para evitar vazamento de informação.

    Returns:
        Série normalizada, média e desvio usados.
    """
    train_size = max(int(series.size(0) * train_ratio), 1)
    train_slice = series[:train_size]
    mean = train_slice.mean().item()
    std = train_slice.std(unbiased=False).item()
    normalized = normalize_with_stats(series, mean, std)
    return normalized, mean, std if std > 1e-12 else 1.0


def create_window_tensors(series: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quebra a série em janelas deslizantes, onde cada janela prevê o próximo valor.

    Returns:
        Tensores (inputs, targets) prontos para alimentar um Dataset.
    """
    series = series.flatten()  # Garante vetor 1-D, evitando dimensões extras acidentais.
    if series.size(0) <= sequence_length:
        raise ValueError("Série insuficiente para criar janelas. Reduza sequence_length.")
    inputs, targets = [], []
    for idx in range(series.size(0) - sequence_length):
        window = series[idx : idx + sequence_length].unsqueeze(-1)
        target = series[idx + sequence_length].unsqueeze(-1)
        inputs.append(window)
        targets.append(target)
    return torch.stack(inputs), torch.stack(targets)


def split_datasets(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float,
) -> Tuple[StockPriceDataset, StockPriceDataset]:
    """Divide as amostras em conjuntos de treino e teste."""
    split_idx = max(int(inputs.size(0) * train_ratio), 1)
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    test_inputs = inputs[split_idx:]
    test_targets = targets[split_idx:]
    if test_inputs.size(0) == 0:
        raise ValueError("Sem exemplos reservados para teste. Ajuste train_ratio.")
    return StockPriceDataset(train_inputs, train_targets), StockPriceDataset(test_inputs, test_targets)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Executa inferência no conjunto informado e retorna métricas + valores desnormalizados."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            preds.append(outputs.cpu())
            trues.append(batch_targets.cpu())
    preds_tensor = torch.cat(preds, dim=0)
    trues_tensor = torch.cat(trues, dim=0)
    # Reverte normalização para métricas em escala real.
    preds_denorm = preds_tensor * std + mean
    trues_denorm = trues_tensor * std + mean
    mae = torch.mean(torch.abs(preds_denorm - trues_denorm)).item()
    rmse = torch.sqrt(torch.mean((preds_denorm - trues_denorm) ** 2)).item()
    mape = (
        torch.mean(torch.abs((trues_denorm - preds_denorm) / trues_denorm.clamp(min=1e-3))) * 100
    ).item()
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return metrics, preds_denorm, trues_denorm


def train(cfg: TrainConfig) -> Dict[str, float]:
    """Fluxo principal de treinamento, avaliação e salvamento."""
    print("Configurações selecionadas para o experimento:")
    for key, value in vars(cfg).items():
        print(f"  - {key}: {value}")
    time.sleep(1.5)

    print("\nBaixando dados históricos...")
    time.sleep(0.5)
    raw_series = download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)
    normalized_series, feature_mean, feature_std = normalize_series(raw_series, cfg.train_ratio)
    inputs, targets = create_window_tensors(normalized_series, cfg.sequence_length)
    print("Preparando conjuntos de treino e teste...")
    time.sleep(0.5)
    train_dataset, test_dataset = split_datasets(inputs, targets, cfg.train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print("\nIniciando treinamento...")
    time.sleep(0.5)
    for epoch in range(1, cfg.epochs + 1):
        print(f"--> Processando época {epoch}/{cfg.epochs}")
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(cfg.device)
            batch_targets = batch_targets.to(cfg.device)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Época {epoch:02d}/{cfg.epochs} - Loss treino: {epoch_loss:.6f}")
        time.sleep(0.2)

    print("\nCalculando métricas no conjunto de teste...")
    time.sleep(0.5)
    metrics, preds_denorm, trues_denorm = evaluate_model(
        model, test_loader, cfg.device, feature_mean, feature_std
    )
    print("\nAvaliação no conjunto de teste (dados reservados):")
    print("Resumo das métricas obtidas:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    time.sleep(1.0)

    print("\nÚltimas previsões vs valores reais:")
    for pred, target in zip(preds_denorm[-5:], trues_denorm[-5:]):
        print(f"Previsto: {pred.item():.2f} | Real: {target.item():.2f}")
    time.sleep(1.0)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    torch.save(checkpoint, cfg.checkpoint_path)
    print(f"\nModelo salvo em {cfg.checkpoint_path}")
    return metrics


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
