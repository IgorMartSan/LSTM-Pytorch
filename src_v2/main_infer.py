from typing import Any, Dict, Optional, Tuple, Sequence

import torch
from torch import nn


class LSTMRegressor(nn.Module):
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
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.regressor(last_hidden)


def load_model(checkpoint_path: str, device: Optional[str] = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]

    if device is None:
        device = cfg.get("device", "cpu")

    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, mean, std, ckpt


def predict_next_from_values(
    values: Sequence[float],
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    model, cfg, mean, std, ckpt = load_model(checkpoint_path, device=device)

    series = torch.tensor(list(values), dtype=torch.float32).flatten()
    mask = torch.isfinite(series)
    cleaned = series[mask]
    if cleaned.numel() == 0:
        raise ValueError("Serie vazia apos remover valores invalidos.")

    seq_len = int(cfg["sequence_length"])
    if cleaned.numel() < seq_len:
        raise ValueError("Serie curta para a janela configurada no checkpoint.")

    window = cleaned[-seq_len:].unsqueeze(0).unsqueeze(-1)
    safe_std = std if std > 1e-12 else 1.0
    window_norm = (window - mean) / safe_std

    device_obj = torch.device(device or cfg.get("device", "cpu"))
    model = model.to(device_obj)
    window_norm = window_norm.to(device_obj)

    with torch.no_grad():
        pred_norm = model(window_norm).cpu()

    pred = pred_norm * std + mean
    return float(pred.squeeze().item()), ckpt


def example_with_fake_data(checkpoint_path: str) -> None:
    values = torch.linspace(10.0, 200.0, steps=200).tolist()
    pred, _ = predict_next_from_values(values, checkpoint_path, device=None)
    print(f"Predicao (serie ficticia): {pred:.6f}")


def tutorial_use_model() -> None:
    """
    Tutorial rapido de uso (para futura API):
    1) Carregar o checkpoint.
    2) Preparar a serie (lista de floats).
    3) Chamar predict_next_from_values.
    """
    checkpoint_path = "best_pareto_lstm.pt"

    model, cfg, _, _, _ = load_model(checkpoint_path, device=None)
    print(f"Modelo carregado. seq_len={cfg['sequence_length']}, device={cfg.get('device', 'cpu')}")
    _ = model

    values = torch.linspace(50.0, 150.0, steps=300).tolist()

    pred, _ = predict_next_from_values(values, checkpoint_path, device=None)
    
    print(f"Predicao (tutorial): {pred:.6f}")


def fastapi_predict_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulacao de endpoint FastAPI para /predict.
    Espera um payload com:
      - values: lista de floats
      - checkpoint_path (opcional): caminho do checkpoint
    """
    values = payload.get("values")
    if values is None:
        raise ValueError("Campo 'values' e obrigatorio.")

    checkpoint_path = payload.get("checkpoint_path", "best_pareto_lstm.pt")
    pred, _ = predict_next_from_values(values, checkpoint_path, device=None)

    return {"prediction": pred}


def main() -> None:
    checkpoint_path = "best_pareto_lstm.pt"
    _ = load_model(checkpoint_path, device=None)
    example_with_fake_data(checkpoint_path)
    tutorial_use_model()


if __name__ == "__main__":
    main()
