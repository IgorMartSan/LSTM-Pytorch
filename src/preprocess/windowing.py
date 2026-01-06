from typing import Tuple
import torch

def create_window_tensors(series: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Série 1D -> janelas:
      inputs: (N, seq_len, 1)
      targets: (N, 1)  (próximo valor após a janela)
    """
    series = series.flatten()
    if series.size(0) <= sequence_length:
        raise ValueError("Série insuficiente para janelas. Reduza sequence_length.")

    inputs, targets = [], []
    for i in range(series.size(0) - sequence_length):
        window = series[i:i + sequence_length].unsqueeze(-1)          # (seq_len, 1)
        target = series[i + sequence_length].unsqueeze(-1)            # (1,)
        inputs.append(window)
        targets.append(target)

    return torch.stack(inputs), torch.stack(targets)

def split_train_val_test(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float,
    val_ratio_within_train: float,
):
    """
    Split temporal: [train][val][test]
    """
    n = inputs.size(0)
    train_end = max(int(n * train_ratio), 1)
    if train_end >= n:
        raise ValueError("Sem exemplos reservados para teste. Ajuste train_ratio.")

    train_inputs_full = inputs[:train_end]
    train_targets_full = targets[:train_end]

    val_size = max(int(train_inputs_full.size(0) * val_ratio_within_train), 1)
    if train_inputs_full.size(0) - val_size < 1:
        raise ValueError("Treino muito pequeno para separar validação.")

    train_inputs = train_inputs_full[:-val_size]
    train_targets = train_targets_full[:-val_size]
    val_inputs = train_inputs_full[-val_size:]
    val_targets = train_targets_full[-val_size:]

    test_inputs = inputs[train_end:]
    test_targets = targets[train_end:]
    if test_inputs.size(0) == 0:
        raise ValueError("Teste vazio. Ajuste train_ratio.")

    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)
