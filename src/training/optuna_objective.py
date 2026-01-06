from dataclasses import asdict
import optuna
import torch

from config import TrainConfig
from preprocess.windowing import create_window_tensors, split_train_val_test
from training.train_one_trial import train_single_trial

def build_objective(cfg: TrainConfig, normalized_series: torch.Tensor, mean: float, std: float):
    def objective(trial: optuna.Trial):
        cfg_local = TrainConfig(**asdict(cfg))

        cfg_local.sequence_length = trial.suggest_int("sequence_length", 20, 120, step=10)
        cfg_local.hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        cfg_local.num_layers = trial.suggest_int("num_layers", 2, 4)
        cfg_local.dropout = 0.3 #trial.suggest_float("dropout", 0.0, 0.3)
        cfg_local.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        cfg_local.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        cfg_local.max_epochs = 60 #trial.suggest_int("max_epochs", 30, 120, step=30)
        cfg_local.patience = 12 #trial.suggest_int("patience", 10, 25, step=5)

        inputs, targets = create_window_tensors(normalized_series, cfg_local.sequence_length)

        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_train_val_test(
            inputs, targets, cfg_local.train_ratio, cfg_local.val_ratio_within_train
        )

        metrics_test, ckpt_path = train_single_trial(
            cfg=cfg_local,
            train_tensors=(tr_x, tr_y),
            val_tensors=(va_x, va_y),
            test_tensors=(te_x, te_y),
            feature_mean=mean,
            feature_std=std,
            trial=trial,
        )

        # multi-objective: MAE, RMSE, MAPE
        return metrics_test["MAE"], metrics_test["RMSE"], metrics_test["MAPE"]

    return objective
