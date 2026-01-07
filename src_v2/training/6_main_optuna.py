import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_optuna = load_module(__file__, "4_optuna_hpo.py", "optuna_hpo")


def main() -> None:
    # Defina hiperparametros e configuracoes aqui
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

    _optuna.run_optuna(
        cfg=cfg,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )


if __name__ == "__main__":
    main()
