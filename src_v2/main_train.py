import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_data = load_module(__file__, "data/1_source_yahoo.py", "data")
_integrity = load_module(__file__, "preprocessing/1_integrity.py", "integrity")
_norm = load_module(__file__, "preprocessing/2_normalization.py", "norm")
_train = load_module(__file__, "training/1_train.py", "train")


def load_data(cfg: TrainConfig):
    return _data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)


def clean_data(series):
    return _integrity.clean_series(series)


def train_and_save(cfg: TrainConfig, series):
    mean, std = _norm.compute_train_stats(series, cfg.train_ratio)
    normalized = _norm.normalize_with_stats(series, mean, std)
    return _train.train_model(
        cfg=cfg,
        normalized_series=normalized,
        feature_mean=mean,
        feature_std=std,
        checkpoint_path=cfg.best_checkpoint_path,
    )


def main() -> None:
    # Defina hiperparametros e configuracoes aqui
    cfg = TrainConfig(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        feature="Close",
        sequence_length=60,
        batch_size=64,
        learning_rate=1e-3,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        max_epochs=120,
        patience=20,
    )

    _train.set_seed(cfg.seed)

    raw = load_data(cfg)
    cleaned, removed = clean_data(raw)
    if removed > 0:
        print(f"Aviso: removidos {removed} valores invalidos da serie.")

    metrics, ckpt_path, val_loss_best = train_and_save(cfg, cleaned)

    print("Treino concluido")
    print(f"checkpoint_path: {ckpt_path}")
    print(f"val_loss_best: {val_loss_best:.6f}")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.6f}")


if __name__ == "__main__":
    main()
