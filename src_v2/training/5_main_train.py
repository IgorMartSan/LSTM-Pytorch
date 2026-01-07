import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_data = load_module(__file__, "../data/1_source_yahoo.py", "data")
_integrity = load_module(__file__, "../preprocessing/1_integrity.py", "integrity")
_norm = load_module(__file__, "../preprocessing/2_normalization.py", "norm")
_train = load_module(__file__, "1_train.py", "train")


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Treino LSTM com pipeline em camadas")

    parser.add_argument("--symbol", default=cfg.symbol)
    parser.add_argument("--start_date", default=cfg.start_date)
    parser.add_argument("--end_date", default=cfg.end_date)
    parser.add_argument("--feature", default=cfg.feature)

    parser.add_argument("--train_ratio", type=float, default=cfg.train_ratio)
    parser.add_argument("--val_ratio_within_train", type=float, default=cfg.val_ratio_within_train)

    parser.add_argument("--sequence_length", type=int, default=cfg.sequence_length)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--hidden_size", type=int, default=cfg.hidden_size)
    parser.add_argument("--num_layers", type=int, default=cfg.num_layers)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)

    parser.add_argument("--max_epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=cfg.patience)

    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--seed", type=int, default=cfg.seed)

    parser.add_argument("--checkpoint_path", default=cfg.best_checkpoint_path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        feature=args.feature,
        train_ratio=args.train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )

    _train.set_seed(cfg.seed)

    raw = _data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)
    cleaned, removed = _integrity.clean_series(raw)

    if removed > 0:
        print(f"Aviso: removidos {removed} valores invalidos da serie.")

    mean, std = _norm.compute_train_stats(cleaned, cfg.train_ratio)
    normalized = _norm.normalize_with_stats(cleaned, mean, std)

    metrics, ckpt_path, val_loss_best = _train.train_model(
        cfg=cfg,
        normalized_series=normalized,
        feature_mean=mean,
        feature_std=std,
        checkpoint_path=args.checkpoint_path,
    )

    print("Treino concluido")
    print(f"checkpoint_path: {ckpt_path}")
    print(f"val_loss_best: {val_loss_best:.6f}")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.6f}")


if __name__ == "__main__":
    main()
