import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_optuna = load_module(__file__, "training/4_optuna_hpo.py", "optuna_hpo")


def load_data(cfg: TrainConfig):
    data = load_module(__file__, "data/1_source_yahoo.py", "data")
    return data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)


def clean_data(series):
    integrity = load_module(__file__, "preprocessing/1_integrity.py", "integrity")
    return integrity.clean_series(series)


def run_optuna_study(cfg: TrainConfig, series, n_trials: int, timeout_sec, best_strategy: str) -> None:
    _optuna.run_optuna_with_series(
        cfg=cfg,
        series=series,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )


def main() -> None:
    cfg = TrainConfig(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        feature="Close",
        train_ratio=0.85,
        val_ratio_within_train=0.15,
        max_epochs=60,
        patience=12,
    )

    n_trials = 40
    timeout_sec = None
    best_strategy = "min_mape"

    raw = load_data(cfg)
    cleaned, removed = clean_data(raw)
    if removed > 0:
        print(f"Aviso: removidos {removed} valores invalidos da serie.")

    run_optuna_study(
        cfg,
        series=cleaned,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )


if __name__ == "__main__":
    main()
