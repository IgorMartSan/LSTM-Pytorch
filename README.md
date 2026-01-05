# Tech Challenge – Fase 04 (Checklist)

## 1. Coleta e Pré-processamento dos Dados (obrigatório)
- [x] Coletar dados históricos de preços de ações de uma empresa escolhida (ex: Yahoo Finance / yfinance)
- [x] Realizar pré-processamento dos dados (limpeza/tratamento)
- [x] Preparar os dados para série temporal (ex: normalização e criação de janelas)
- [x] Separar dados para treino e validação/teste

## 2. Desenvolvimento do Modelo LSTM (obrigatório)
- [ ] Implementar um modelo LSTM para prever o valor de fechamento (Close)
- [ ] Treinar o modelo e ajustar hiperparâmetros para melhorar desempenho
- [ ] Avaliar o modelo com métricas (MAE, RMSE, MAPE ou outra apropriada)

## 3. Salvamento e Exportação do Modelo (obrigatório)
- [ ] Salvar o modelo treinado em formato adequado para inferência

## 4. Deploy do Modelo em API (obrigatório)
- [ ] Criar uma API REST usando Flask ou FastAPI para servir o modelo
- [ ] Implementar endpoint que receba dados históricos e retorne previsões de preços futuros

## 5. Escalabilidade e Monitoramento (obrigatório)
- [ ] Configurar monitoramento para performance do modelo em produção (tempo de resposta e uso de recursos)

## Entregáveis (obrigatório)
- [ ] Código-fonte do modelo LSTM no repositório Git com documentação
- [ ] Scripts ou contêineres Docker para deploy da API
- [ ] Link da API em produção (se houver deploy em nuvem)
- [ ] Vídeo mostrando e explicando o funcionamento da API



optuna-dashboard sqlite:///optuna_study.db
