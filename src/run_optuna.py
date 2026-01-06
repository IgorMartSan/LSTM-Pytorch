import optuna
import torch

from config import TrainConfig
from utils.seed import set_seed
from utils.logging_utils import setup_logger
from data.yahoo import download_price_series
from preprocess.normalization import compute_train_stats, normalize_with_stats
from training.pareto import export_pareto_csv, choose_best_from_pareto
from training.checkpointing import load_checkpoint
from training.optuna_objective import build_objective

logger = setup_logger()

def run_optuna(cfg: TrainConfig, n_trials: int = 30, timeout_sec=None, best_strategy: str = "min_mape") -> None:
    set_seed(cfg.seed)

    logger.info("Baixando dados Yahoo...")
    raw = download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)

    logger.info("Normalização sem vazamento (stats só do bloco de treino)...")
    mean, std = compute_train_stats(raw, cfg.train_ratio)
    normalized = normalize_with_stats(raw, mean, std)

    storage = optuna.storages.RDBStorage(url=cfg.storage_path)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],
        pruner=pruner,
    )
    study.set_metric_names(["MAE", "RMSE", "MAPE"])

    objective = build_objective(cfg, normalized, mean, std)

    logger.info(f"Iniciando Optuna: trials={n_trials}, timeout={timeout_sec}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    export_pareto_csv(study, cfg.pareto_csv_path)
    logger.info(f"Pareto exportado em: {cfg.pareto_csv_path}")

    pareto = study.best_trials
    best_trial = choose_best_from_pareto(pareto, strategy=best_strategy)

    logger.info("===== MELHOR TRIAL (Pareto) =====")
    logger.info(f"trial_number: {best_trial.number}")
    logger.info(f"MAE : {best_trial.values[0]:.6f}")
    logger.info(f"RMSE: {best_trial.values[1]:.6f}")
    logger.info(f"MAPE: {best_trial.values[2]:.6f}")
    logger.info(f"val_loss_best: {best_trial.user_attrs.get('val_loss_best')}")
    logger.info(f"checkpoint_path: {best_trial.user_attrs.get('checkpoint_path')}")

    ckpt_path = best_trial.user_attrs.get("checkpoint_path")
    if not ckpt_path:
        logger.warning("Não encontrei checkpoint_path no best_trial.user_attrs (talvez trials antigos).")
        return

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    torch.save(ckpt, cfg.best_checkpoint_path)
    logger.info(f"Checkpoint 'best' salvo em: {cfg.best_checkpoint_path}")

if __name__ == "__main__":
    cfg = TrainConfig(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        feature="Close",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    run_optuna(cfg, n_trials=60, timeout_sec=None, best_strategy="min_mape")
