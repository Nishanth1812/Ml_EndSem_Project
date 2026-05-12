from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["baseline", "random_forest", "xgboost"]


class StatusResponse(BaseModel):
    status: str
    data_path: str
    trained: bool


class TrainResponse(BaseModel):
    message: str
    metrics: dict[str, dict[str, float]]


class DataResponse(BaseModel):
    rows: int
    columns: list[str]
    preview: list[dict[str, object]]


class MetricsResponse(BaseModel):
    metrics: dict[str, dict[str, float]]


class PredictRequest(BaseModel):
    model_name: ModelName = "xgboost"


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    model_name: ModelName = "xgboost"
    lookback_days: int = Field(default=365, ge=60, le=3650)


class RankRequest(BaseModel):
    model_name: ModelName = "xgboost"
    top_n: int = Field(default=5, ge=1, le=15)
    tickers: list[str] | None = None
    lookback_days: int = Field(default=365, ge=60, le=3650)


class AnalysisResponse(BaseModel):
    ticker: str
    model_name: str
    current_price: float
    predicted_return: float
    market_expected_return: float
    predicted_alpha: float
    beta: float
    volatility: float
    rsi: float


class RankItem(BaseModel):
    ticker: str
    model_name: str
    current_price: float
    predicted_return: float
    market_expected_return: float
    predicted_alpha: float
    beta: float
    volatility: float
    rsi: float


class RankResponse(BaseModel):
    items: list[RankItem]
