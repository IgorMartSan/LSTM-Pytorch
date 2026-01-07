# README de Explicacao (para roteiro do video)

Este documento explica, em linguagem simples, cada parte do codigo do projeto e como usar os scripts.

## 1) Visao geral do projeto

O objetivo e treinar um modelo LSTM para prever o valor de fechamento (Close) de uma acao a partir de series temporais.
O pipeline foi organizado em camadas para separar responsabilidades:

- **1_data**: busca os dados na fonte (Yahoo Finance)
- **2_preprocessing**: limpa dados nulos/invalidos, normaliza e cria janelas temporais
- **3_model**: define a arquitetura do LSTM
- **3_training**: treina, avalia, salva checkpoints e executa Optuna
- **main_***: pontos de entrada (scripts) para rodar treino, Optuna e inferencia

## 2) Estrutura de pastas (src_v2)

- `src_v2/data/1_source_yahoo.py`
  - Faz download dos dados de mercado com `yfinance`.
  - Retorna a serie de precos como `torch.Tensor`.

- `src_v2/preprocessing/1_integrity.py`
  - Remove valores invalidos (NaN, inf) e verifica se a serie ficou valida.

- `src_v2/preprocessing/2_normalization.py`
  - Calcula media e desvio do trecho de treino.
  - Normaliza a serie usando esses valores para evitar vazamento de dados.
  - Tambem faz a desnormalizacao na hora de avaliar.

- `src_v2/preprocessing/3_windowing.py`
  - Transforma a serie 1D em janelas (X) e alvos (y).
  - Faz o split temporal em treino, validacao e teste.

- `src_v2/model/1_lstm.py`
  - Define o modelo `LSTMRegressor` com camadas LSTM + Linear.
  - A previsao e feita usando o ultimo estado oculto da LSTM.

- `src_v2/training/1_train.py`
  - Faz o treino da LSTM com MSE.
  - Aplica early stopping usando `patience`.
  - Salva checkpoint com pesos, config, metricas e estatisticas.

- `src_v2/training/2_evaluate.py`
  - Calcula metricas no conjunto de teste (em escala real, desnormalizada).

- `src_v2/training/3_checkpointing.py`
  - Salva e carrega checkpoints.

- `src_v2/training/4_optuna_hpo.py`
  - Cria o estudo Optuna, define o espaco de busca e roda os trials.
  - Exporta os melhores trials (Pareto) para CSV.

- `src_v2/training/8_inference.py`
  - Carrega o modelo a partir do checkpoint.
  - Faz previsao do proximo ponto usando a ultima janela.

## 3) Mains (pontos de entrada)

- `src_v2/main_train.py`
  - Executa o pipeline de treino completo (dados -> limpeza -> normalizacao -> treino -> salva checkpoint).
  - Ideal para treinar com hiperparametros fixos.

- `src_v2/main_optuna.py`
  - Executa o pipeline com Optuna para buscar hiperparametros.
  - Parametros do estudo (n_trials, timeout, estrategia) ficam neste arquivo.

- `src_v2/main_infer.py`
  - Exemplo simples de inferencia.
  - Carrega o modelo treinado e faz previsao com dados ficticios.

## 4) Hiperparametros (onde alterar)

- **Treino simples**: edite em `src_v2/main_train.py` (sequence_length, batch_size, learning_rate, hidden_size, etc.)
- **Optuna**:
  - Parametros do estudo (n_trials, timeout, best_strategy) em `src_v2/main_optuna.py`
  - Espaco de busca em `src_v2/training/4_optuna_hpo.py` (função `build_objective`)

## 5) Como funciona o treino (explicacao detalhada)

O treino acontece em `src_v2/training/1_train.py` e segue o fluxo abaixo:

- **Modelo (LSTM + Linear)**\n  - A LSTM recebe a janela de entrada (seq_len x 1).\n  - A saida usada para previsao e o ultimo estado oculto.\n  - Uma camada Linear transforma esse estado em um valor unico (previsao do proximo ponto).\n+
- **Funcao de ativacao**\n  - A LSTM ja possui funcoes internas (sigmoid/tanh) para controlar os gates.\n  - A camada Linear final nao aplica ativacao (saida continua), porque o problema e de regressao.\n+
- **Funcao de loss (erro)**\n  - E usado MSE (Mean Squared Error).\n  - O MSE penaliza mais os erros grandes e ajuda a estabilizar o treino.\n+
- **Otimizador**\n  - E usado Adam com `learning_rate` configurado no `TrainConfig`.\n  - Adam combina momento + adaptacao de taxa para cada parametro.\n+
- **Early stopping (patience)**\n  - Apos cada epoca, calcula-se o loss de validacao.\n  - Se o loss nao melhora por `patience` epocas, o treino para.\n+
- **Checkpoint**\n  - O melhor modelo (menor loss de validacao) e salvo com pesos e configuracoes.\n+  - O arquivo guarda: pesos, metricas, media/desvio usados na normalizacao e hiperparametros.

## 6) Metricas usadas

As metricas sao calculadas no conjunto de teste, usando valores desnormalizados:

- **MAE (Mean Absolute Error)**\n  - Media do erro absoluto: soma dos |erro| dividido pelo numero de amostras.\n  - Unidades iguais ao preco (ex: reais/dolares).\n  - Quanto menor, melhor. Mede erro medio sem penalizar muito outliers.\n+
- **RMSE (Root Mean Squared Error)**\n  - Raiz quadrada do erro quadratico medio.\n  - Penaliza erros grandes com mais peso (porque eleva o erro ao quadrado).\n  - Quanto menor, melhor. Sensivel a outliers.\n+
- **MAPE (Mean Absolute Percentage Error)**\n  - Erro percentual medio absoluto.\n  - Exemplo: MAPE = 5% significa que, em media, o erro foi 5% do valor real.\n  - Quanto menor, melhor. Pode ser instavel quando o valor real e muito pequeno.

Estas tres metricas sao usadas no Optuna como multi-objetivo.

## 7) Como executar

Treino simples:

```
python src_v2/main_train.py
```

Optuna:

```
python src_v2/main_optuna.py
```

Inferencia:

```
python src_v2/main_infer.py
```

## 8) Explicacao do fluxo para o video

1. **Coleta**: o script baixa dados de precos do Yahoo Finance.
2. **Limpeza**: remove valores invalidos para evitar erros.
3. **Normalizacao**: calcula media/desvio do treino e normaliza a serie.
4. **Janelamento**: cria sequencias temporais para a LSTM aprender.
5. **Treino**: a LSTM aprende a prever o proximo valor.
6. **Avaliacao**: calcula MAE, RMSE e MAPE em escala real.
7. **Salvamento**: salva checkpoint com pesos e configuracoes.
8. **Optuna (opcional)**: busca hiperparametros automaticamente.
9. **Inferencia**: carrega o checkpoint e gera uma previsao.

## 9) Observacoes finais

- Se precisar alterar a origem dos dados, edite `src_v2/data/1_source_yahoo.py`.
- Se quiser adicionar novas features, o ideal e expandir a serie para tensores com mais colunas.
- Para deploy em API, voce pode reutilizar o arquivo de inferencia como base.
