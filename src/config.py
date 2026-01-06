from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    feature: str = "Close"

    # Split temporal
    train_ratio: float = 0.85
    val_ratio_within_train: float = 0.15

    # Defaults (Optuna sobrescreve)
    sequence_length: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # Treino
    max_epochs: int = 120
    patience: int = 20

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Optuna
    study_name: str = "stock_lstm_optuna"
    storage_path: str = "sqlite:///optuna_study.db"
    pareto_csv_path: str = "pareto_trials.csv"
    best_checkpoint_path: str = "best_pareto_lstm.pt"
    checkpoints_dir: str = "checkpoints"
