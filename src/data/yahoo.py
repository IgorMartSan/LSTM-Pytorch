import torch
import yfinance as yf

def download_price_series(symbol: str, start_date: str, end_date: str, feature: str) -> torch.Tensor:
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError("Dataset retornou vazio. Verifique símbolo e datas.")
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' não encontrada. Colunas: {list(df.columns)}")
    values = torch.tensor(df[feature].values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Série retornou vazia após conversão.")
    return values
