"""
Script de inferência que utiliza o modelo salvo em train_lstm.py para gerar previsões
no conjunto de teste reservado.
"""

import time

import torch
from torch.utils.data import DataLoader

from train_lstm import (
    LSTMRegressor,
    TrainConfig,
    download_price_series,
    normalize_with_stats,
    create_window_tensors,
    split_datasets,
    evaluate_model,
)


def run_inference(checkpoint_path: str = "stock_lstm_checkpoint.pt", samples_to_show: int = 5) -> None:
    """Carrega o checkpoint, recompõe o dataset de teste e exibe previsões."""
    print("Carregando checkpoint salvo...")
    time.sleep(0.5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainConfig(**checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Reconstroi dataset com mesma normalização usada no treino.
    print("Preparando dados normalizados para inferência...")
    time.sleep(0.5)
    raw_series = download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)
    normalized_series = normalize_with_stats(
        raw_series, checkpoint["feature_mean"], checkpoint["feature_std"]
    )
    inputs, targets = create_window_tensors(normalized_series, cfg.sequence_length)
    _, test_dataset = split_datasets(inputs, targets, cfg.train_ratio)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    print("Gerando previsões no conjunto de teste reservado...")
    time.sleep(0.5)
    metrics, preds_denorm, trues_denorm = evaluate_model(
        model, test_loader, device, checkpoint["feature_mean"], checkpoint["feature_std"]
    )

    print("\nMétricas obtidas na inferência:")
    time.sleep(0.5)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    time.sleep(1.0)

    print("\nAlguns exemplos de previsões:")
    for pred, target in zip(preds_denorm[-samples_to_show:], trues_denorm[-samples_to_show:]):
        print(f"Previsto: {pred.item():.2f} | Real: {target.item():.2f}")


if __name__ == "__main__":
    run_inference()
