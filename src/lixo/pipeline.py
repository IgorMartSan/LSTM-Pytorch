"""Fluxo interativo para baixar histórico completo de um ticker e salvar em CSV."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import yfinance as yf

def wait_for_enter(message: str) -> None:
    """Pausa a execução até o usuário pressionar Enter."""
    input(message)


def create_windows(series: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera pares (janela, próximo valor) a partir da série normalizada."""

    if series.size <= sequence_length:
        raise ValueError(
            "Série insuficiente para gerar janelas. Reduza 'sequence_length' ou colete mais dados."
        )
    inputs = []
    targets = []
    for idx in range(series.size - sequence_length):
        inputs.append(series[idx : idx + sequence_length])
        targets.append(series[idx + sequence_length])
    return np.stack(inputs), np.stack(targets)


def split_windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide janelas em conjuntos de treino/val/teste preservando ordem temporal."""

    total = inputs.shape[0]
    if total < 3:
        raise ValueError("São necessárias ao menos 3 janelas para realizar a divisão.")
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    train_end = max(int(total * train_ratio), 1)
    val_end = train_end + max(int(total * val_ratio), 1)
    if val_end >= total:
        val_end = total - 1
    if train_end >= val_end:
        raise ValueError("As proporções selecionadas não deixam exemplos suficientes para validação.")
    if total - val_end <= 0:
        raise ValueError("As proporções selecionadas não deixam exemplos para teste.")
    train_inputs = inputs[:train_end]
    train_targets = targets[:train_end]
    val_inputs = inputs[train_end:val_end]
    val_targets = targets[train_end:val_end]
    test_inputs = inputs[val_end:]
    test_targets = targets[val_end:]
    return train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets


def main() -> None:
    print("=== Downloader de histórico máximo de ativos ===")

    symbol = "BTC-USD"
    output_dir = Path("data")
    output_file = output_dir / f"{symbol}_historico_completo.csv"
    feature_column = "Close"
    sequence_length = 60
    train_ratio = 0.7
    val_ratio = 0.15

    wait_for_enter("\nPressione Enter para iniciar a etapa de download dos dados...")

    # ETAPA 1 - BAIXANDO OS DADOS
    try:
        dataset = yf.download(symbol, period="max", interval="1d", progress=True)
    except Exception as exc:  # noqa: BLE001 - encerramos execução com mensagem simples
        sys.exit(f"Falha ao baixar dados de {symbol}: {exc}")

    if dataset.empty:
        sys.exit("Nenhum dado retornado. Verifique se o ticker existe e tente novamente.")

    first_date = dataset.index.min().date()
    last_date = dataset.index.max().date()
    print(
        f"\nDownload concluído com {len(dataset)} registros de {first_date} até {last_date}."
    )

    wait_for_enter("\nPressione Enter para iniciar a verificação de dados problemáticos...")

    # ETAPA 2 - RELATÓRIO DE QUALIDADE DOS DADOS
    missing_counts = dataset.isna().sum()
    print("\nValores ausentes por coluna:")
    print(missing_counts)
    problematic_missing = missing_counts[missing_counts > 0]
    if not problematic_missing.empty:
        print("\nAviso: colunas com valores ausentes detectados:")
        for column, count in problematic_missing.items():
            print(f"  - {column}: {count} registros ausentes")
    else:
        print("\nNenhum valor ausente encontrado.")

    numeric_columns = dataset.select_dtypes(include=[np.number])
    if not numeric_columns.empty:
        inf_counts = np.isinf(numeric_columns).sum()
        problematic_infs = inf_counts[inf_counts > 0]
        if not problematic_infs.empty:
            print("\nAviso: valores infinitos detectados:")
            for column, count in problematic_infs.items():
                print(f"  - {column}: {count} registros +/- infinito")
        else:
            print("\nNenhum valor infinito encontrado.")
    else:
        print("\nNão há colunas numéricas para verificar infinitos.")

    duplicate_count = dataset.index.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nAviso: {duplicate_count} datas duplicadas detectadas no índice.")
    else:
        print("\nNenhuma data duplicada detectada.")

    print("\nEstatísticas descritivas (pandas describe):")
    print(dataset.describe().transpose())

    wait_for_enter("\nPressione Enter para salvar o CSV e concluir o processo...")

    # ETAPA 3 - SALVANDO OS DADOS
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file)
    print(f"\nArquivo salvo em: {output_file.resolve()}")

    wait_for_enter("\nPressione Enter para preparar os dados para série temporal...")

    # ETAPA 4 - PREPARAÇÃO PARA SÉRIE TEMPORAL

    if feature_column not in dataset.columns:
        sys.exit(
            f"A coluna '{feature_column}' não está presente no dataset. "
            f"Colunas disponíveis: {list(dataset.columns)}"
        )

    feature_series = dataset[feature_column].dropna().astype("float32").to_numpy()
    if feature_series.size == 0:
        sys.exit("A coluna selecionada não possui valores numéricos válidos.")

    train_size = max(int(feature_series.size * train_ratio), 1)
    mean = feature_series[:train_size].mean()
    std = feature_series[:train_size].std()
    safe_std = std if std > 1e-12 else 1.0
    normalized = (feature_series - mean) / safe_std

    print(
        f"\nNormalização concluída usando mean={mean:.4f} e std={safe_std:.4f} "
        f"(calculados sobre {train_size} registros de treino)."
    )

    inputs, targets = create_windows(normalized, sequence_length)
    print(f"Foram criadas {inputs.shape[0]} janelas de tamanho {sequence_length}.")

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        test_inputs,
        test_targets,
    ) = split_windows(inputs, targets, train_ratio=train_ratio, val_ratio=val_ratio)

    print(
        "Divisão concluída -> "
        f"treino: {train_inputs.shape[0]} | validação: {val_inputs.shape[0]} | teste: {test_inputs.shape[0]}"
    )

    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / f"{symbol}_windows.npz"
    np.savez_compressed(
        
        processed_file,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        feature_mean=mean,
        feature_std=safe_std,
        sequence_length=sequence_length,
        feature_column=feature_column,
    )
    print(f"Arrays de séries temporais salvos em: {processed_file.resolve()}")


if __name__ == "__main__":
    main()
