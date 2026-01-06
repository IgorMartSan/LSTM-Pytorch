from typing import List
import optuna
import pandas as pd

def choose_best_from_pareto(pareto_trials: List[optuna.trial.FrozenTrial], strategy: str = "min_mape"):
    if not pareto_trials:
        raise ValueError("Pareto vazio.")

    if strategy == "min_mae":
        return min(pareto_trials, key=lambda t: t.values[0])
    if strategy == "min_rmse":
        return min(pareto_trials, key=lambda t: t.values[1])
    if strategy == "min_mape":
        return min(pareto_trials, key=lambda t: t.values[2])

    if strategy == "weighted":
        w_mae, w_rmse, w_mape = 1.0, 1.0, 1.0
        return min(pareto_trials, key=lambda t: w_mae*t.values[0] + w_rmse*t.values[1] + w_mape*t.values[2])

    raise ValueError(f"Estratégia desconhecida: {strategy}")

def export_pareto_csv(study: optuna.Study, path: str) -> None:
    rows = []
    for t in study.best_trials:
        rows.append({
            "trial_number": t.number,
            "MAE": t.values[0],
            "RMSE": t.values[1],
            "MAPE": t.values[2],
            "val_loss_best": t.user_attrs.get("val_loss_best"),
            "checkpoint_path": t.user_attrs.get("checkpoint_path"),
            "params": t.params,
        })
    df = pd.DataFrame(rows).sort_values(by="MAPE", ascending=True)
    df.to_csv(path, index=False)
