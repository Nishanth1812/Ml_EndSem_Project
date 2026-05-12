from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd

from src.alpha_engine.engine import DEFAULT_WATCHLIST, StockAlphaEngine


class EngineService:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self._lock = Lock()
        self._engine = StockAlphaEngine(self.data_path)

    @property
    def engine(self) -> StockAlphaEngine:
        return self._engine

    def status(self) -> dict[str, object]:
        return {
            "status": "ok",
            "data_path": str(self.data_path),
            "trained": self._engine.is_trained,
        }

    def load_data(self) -> pd.DataFrame:
        return self._engine.prepare_data()

    def train(self) -> dict[str, object]:
        with self._lock:
            self._engine.train()
            return self._engine.metrics_dataframe().to_dict(orient="index")

    def metrics(self) -> dict[str, object]:
        return self._engine.metrics_dataframe().to_dict(orient="index")

    def analyze(self, ticker: str, model_name: str = "xgboost", lookback_days: int = 365):
        return self._engine.analyze_ticker(ticker, model_name=model_name, lookback_days=lookback_days)

    def rank(self, tickers: list[str] | None = None, model_name: str = "xgboost", top_n: int = 5, lookback_days: int = 365):
        ticker_list = tickers if tickers else DEFAULT_WATCHLIST
        return self._engine.rank_tickers(ticker_list, model_name=model_name, top_n=top_n, lookback_days=lookback_days)
