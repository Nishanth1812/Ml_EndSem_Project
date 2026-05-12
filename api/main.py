from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    DataResponse,
    MetricsResponse,
    RankItem,
    RankRequest,
    RankResponse,
    StatusResponse,
    TrainResponse,
)
from .service import EngineService

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "NIFTY 500_day.csv"
service = EngineService(DATA_PATH)

app = FastAPI(title="Alpha Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=StatusResponse)
def health() -> StatusResponse:
    return StatusResponse(**service.status())


@app.get("/data", response_model=DataResponse)
def get_data() -> DataResponse:
    try:
        frame = service.load_data()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DataResponse(
        rows=len(frame),
        columns=list(frame.columns),
        preview=frame.tail(5).reset_index().to_dict(orient="records"),
    )


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    try:
        metrics = service.train()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrainResponse(message="Models trained successfully.", metrics=metrics)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        metrics_map = service.metrics()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MetricsResponse(metrics=metrics_map)


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: AnalyzeRequest) -> AnalysisResponse:
    try:
        analysis = service.analyze(payload.ticker, payload.model_name, payload.lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisResponse(
        ticker=analysis.ticker,
        model_name=analysis.model_name,
        current_price=analysis.current_price,
        predicted_return=analysis.predicted_return,
        market_expected_return=analysis.market_expected_return,
        predicted_alpha=analysis.predicted_alpha,
        beta=analysis.beta,
        volatility=analysis.volatility,
        rsi=analysis.rsi,
    )


@app.post("/rank", response_model=RankResponse)
def rank(payload: RankRequest) -> RankResponse:
    try:
        items = service.rank(payload.tickers, payload.model_name, payload.top_n, payload.lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RankResponse(
        items=[
            RankItem(
                ticker=item.ticker,
                model_name=item.model_name,
                current_price=item.current_price,
                predicted_return=item.predicted_return,
                market_expected_return=item.market_expected_return,
                predicted_alpha=item.predicted_alpha,
                beta=item.beta,
                volatility=item.volatility,
                rsi=item.rsi,
            )
            for item in items
        ]
    )
