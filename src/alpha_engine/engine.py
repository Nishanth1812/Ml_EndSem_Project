from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ModelName = Literal["baseline", "random_forest", "xgboost"]

DEFAULT_WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "BHARTIARTL.NS",
    "HCLTECH.NS",
    "ASIANPAINT.NS",
]

FEATURE_COLUMNS = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "price_to_sma_5",
    "price_to_sma_10",
    "price_to_sma_20",
    "price_to_sma_50",
    "momentum_5",
    "momentum_10",
    "volatility_10",
    "volatility_21",
    "rsi_14",
    "volume_change_5",
    "volume_ratio_20",
]


@dataclass(slots=True)
class ModelMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass(slots=True)
class StockAnalysis:
    ticker: str
    model_name: str
    current_price: float
    predicted_return: float
    market_expected_return: float
    predicted_alpha: float
    beta: float
    volatility: float
    rsi: float
    data: pd.DataFrame


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(column[0]) for column in frame.columns]
    return frame


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _flatten_columns(frame.copy())

    if "date" not in {str(column).strip().lower() for column in normalized.columns}:
        normalized = normalized.reset_index()

    rename_map = {}
    for column in normalized.columns:
        key = str(column).strip().lower().replace(" ", "_")
        if key in {"date", "datetime", "timestamp", "index"}:
            rename_map[column] = "date"
        elif key in {"open", "high", "low", "close", "adj_close", "volume"}:
            rename_map[column] = "close" if key == "adj_close" else key

    normalized = normalized.rename(columns=rename_map)

    if "date" not in normalized.columns:
        raise ValueError("Price history requires a date column.")
    if "close" not in normalized.columns:
        raise ValueError("Price history requires a close column.")
    if "volume" not in normalized.columns:
        normalized["volume"] = 1.0

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(1.0)
    normalized = normalized.dropna(subset=["date", "close"]).sort_values("date").set_index("date")
    normalized["volume"] = normalized["volume"].replace(0, 1.0)
    return normalized


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50.0).clip(lower=0, upper=100)


class StockAlphaEngine:
    def __init__(self, data_path: str | Path, market_ticker: str = "^NSEI", random_state: int = 42):
        self.data_path = Path(data_path)
        self.market_ticker = market_ticker
        self.random_state = random_state

        self.raw_frame: pd.DataFrame | None = None
        self.feature_frame: pd.DataFrame | None = None
        self.holdout_frame: pd.DataFrame | None = None
        self.metrics_: dict[str, ModelMetrics] = {}

        self.baseline_model: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        self.random_forest_model: RandomForestRegressor | None = None
        self.xgboost_model: XGBRegressor | None = None
        self.is_trained = False

    def load_price_history(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return _normalize_price_frame(pd.read_csv(self.data_path))

    def _build_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        close = enriched["close"].astype(float)
        volume = enriched["volume"].astype(float)

        enriched["returns"] = np.log(close / close.shift(1))
        for lag in (1, 2, 3, 5):
            enriched[f"return_lag_{lag}"] = enriched["returns"].shift(lag)

        for window in (5, 10, 20, 50):
            sma = close.rolling(window=window, min_periods=window).mean()
            enriched[f"sma_{window}"] = sma
            enriched[f"price_to_sma_{window}"] = close / sma

        enriched["momentum_5"] = close.pct_change(5)
        enriched["momentum_10"] = close.pct_change(10)
        enriched["volatility_10"] = enriched["returns"].rolling(window=10, min_periods=10).std()
        enriched["volatility_21"] = enriched["returns"].rolling(window=21, min_periods=21).std()
        enriched["rsi_14"] = _calculate_rsi(close, window=14)
        enriched["volume_change_5"] = volume.pct_change(5).replace([np.inf, -np.inf], np.nan)
        enriched["volume_ratio_20"] = volume / volume.rolling(window=20, min_periods=20).mean()
        enriched["target_next_return"] = enriched["returns"].shift(-1)

        enriched = enriched.replace([np.inf, -np.inf], np.nan)
        enriched = enriched.dropna(subset=FEATURE_COLUMNS + ["target_next_return"]).copy()
        return enriched

    def prepare_data(self) -> pd.DataFrame:
        self.raw_frame = self.load_price_history()
        self.feature_frame = self._build_feature_frame(self.raw_frame)
        return self.feature_frame

    def _time_series_cv(self, sample_count: int) -> TimeSeriesSplit:
        if sample_count < 60:
            return TimeSeriesSplit(n_splits=2)
        if sample_count < 120:
            return TimeSeriesSplit(n_splits=3)
        if sample_count < 240:
            return TimeSeriesSplit(n_splits=4)
        return TimeSeriesSplit(n_splits=5)

    def _fit_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            param_distributions={
                "n_estimators": [150, 250, 400],
                "max_depth": [None, 4, 6, 8],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.7, 1.0],
            },
            n_iter=6,
            scoring="neg_mean_absolute_error",
            cv=self._time_series_cv(len(X_train)),
            random_state=self.random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    def _fit_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
        search = RandomizedSearchCV(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="rmse",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            param_distributions={
                "n_estimators": [150, 250, 350],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.02, 0.05, 0.08],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [1.0, 2.0, 5.0],
            },
            n_iter=6,
            scoring="neg_mean_absolute_error",
            cv=self._time_series_cv(len(X_train)),
            random_state=self.random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    @staticmethod
    def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> ModelMetrics:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return ModelMetrics(mae=mae, rmse=rmse, r2=r2)

    def train(self) -> dict[str, ModelMetrics]:
        frame = self.feature_frame if self.feature_frame is not None else self.prepare_data()
        if len(frame) < 100:
            raise ValueError("The dataset is too small to train the models reliably.")

        split_index = int(len(frame) * 0.8)
        if split_index < 30 or len(frame) - split_index < 10:
            raise ValueError("Not enough data points for a chronological train/test split.")

        train_frame = frame.iloc[:split_index].copy()
        test_frame = frame.iloc[split_index:].copy()

        X_train = train_frame[FEATURE_COLUMNS]
        y_train = train_frame["target_next_return"]
        X_test = test_frame[FEATURE_COLUMNS]
        y_test = test_frame["target_next_return"]

        self.baseline_model.fit(X_train, y_train)
        self.random_forest_model = self._fit_random_forest(X_train, y_train)
        self.xgboost_model = self._fit_xgboost(X_train, y_train)

        baseline_pred = self.baseline_model.predict(X_test)
        rf_pred = self.random_forest_model.predict(X_test)
        xgb_pred = self.xgboost_model.predict(X_test)

        self.metrics_ = {
            "baseline": self._calculate_metrics(y_test, baseline_pred),
            "random_forest": self._calculate_metrics(y_test, rf_pred),
            "xgboost": self._calculate_metrics(y_test, xgb_pred),
        }

        self.holdout_frame = test_frame.copy()
        self.holdout_frame["actual_next_return"] = y_test.values
        self.holdout_frame["pred_baseline"] = baseline_pred
        self.holdout_frame["pred_random_forest"] = rf_pred
        self.holdout_frame["pred_xgboost"] = xgb_pred
        self.is_trained = True
        return self.metrics_

    def metrics_dataframe(self) -> pd.DataFrame:
        if not self.metrics_:
            raise ValueError("Train the models before requesting metrics.")

        rows = []
        for model_name, metrics in self.metrics_.items():
            rows.append(
                {
                    "model": model_name,
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                    "r2": metrics.r2,
                }
            )
        return pd.DataFrame(rows).set_index("model").sort_values("mae")

    def _get_model(self, model_name: ModelName):
        if model_name == "baseline":
            return self.baseline_model
        if model_name == "random_forest":
            if self.random_forest_model is None:
                raise ValueError("Random forest model is not trained yet.")
            return self.random_forest_model
        if model_name == "xgboost":
            if self.xgboost_model is None:
                raise ValueError("XGBoost model is not trained yet.")
            return self.xgboost_model
        raise ValueError(f"Unsupported model: {model_name}")

    def predict(self, frame: pd.DataFrame, model_name: ModelName = "xgboost") -> np.ndarray:
        model = self._get_model(model_name)
        features = frame[FEATURE_COLUMNS]
        return np.asarray(model.predict(features))

    def feature_importance(self, model_name: ModelName = "xgboost") -> pd.Series:
        if model_name == "baseline":
            coefficients = np.abs(self.baseline_model.named_steps["regressor"].coef_)
            return pd.Series(coefficients, index=FEATURE_COLUMNS).sort_values(ascending=False)
        if model_name == "random_forest":
            if self.random_forest_model is None:
                raise ValueError("Random forest model is not trained yet.")
            return pd.Series(self.random_forest_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        if model_name == "xgboost":
            if self.xgboost_model is None:
                raise ValueError("XGBoost model is not trained yet.")
            return pd.Series(self.xgboost_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        raise ValueError(f"Unsupported model: {model_name}")

    def download_ticker_history(self, ticker: str, lookback_days: int = 365) -> pd.DataFrame:
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if ticker_data.empty:
            raise ValueError(f"No price history found for {ticker}.")
        return _normalize_price_frame(ticker_data)

    def estimate_beta(self, stock_frame: pd.DataFrame, market_frame: pd.DataFrame) -> float:
        combined = pd.concat(
            [stock_frame["returns"].rename("stock"), market_frame["returns"].rename("market")],
            axis=1,
            join="inner",
        ).dropna()
        if len(combined) < 20:
            return 1.0

        market_variance = float(combined["market"].var())
        if market_variance <= 0 or np.isnan(market_variance):
            return 1.0

        beta = float(combined["stock"].cov(combined["market"]) / market_variance)
        return float(np.clip(beta, 0.1, 3.0))

    def analyze_ticker(self, ticker: str, model_name: ModelName = "xgboost", lookback_days: int = 365) -> StockAnalysis:
        if not self.is_trained:
            raise ValueError("Train the models before running ticker inference.")

        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker.startswith("^") and not normalized_ticker.endswith(".NS"):
            normalized_ticker = f"{normalized_ticker}.NS"

        stock_frame = self._build_feature_frame(self.download_ticker_history(normalized_ticker, lookback_days))
        if stock_frame.empty:
            raise ValueError(f"Not enough recent history to score {normalized_ticker}.")

        market_frame = self._build_feature_frame(self.download_ticker_history(self.market_ticker, lookback_days))

        latest_row = stock_frame.tail(1)
        predicted_return = float(self.predict(latest_row, model_name=model_name)[0])
        beta = self.estimate_beta(stock_frame, market_frame)
        market_expected_return = float(market_frame["returns"].tail(20).mean())
        predicted_alpha = float(predicted_return - (beta * market_expected_return))

        return StockAnalysis(
            ticker=normalized_ticker,
            model_name=model_name,
            current_price=float(stock_frame["close"].iloc[-1]),
            predicted_return=predicted_return,
            market_expected_return=market_expected_return,
            predicted_alpha=predicted_alpha,
            beta=beta,
            volatility=float(stock_frame["volatility_21"].iloc[-1]),
            rsi=float(stock_frame["rsi_14"].iloc[-1]),
            data=stock_frame,
        )

    def rank_tickers(
        self,
        tickers: list[str],
        model_name: ModelName = "xgboost",
        top_n: int = 5,
        lookback_days: int = 365,
    ) -> list[StockAnalysis]:
        analyses: list[StockAnalysis] = []
        for ticker in tickers:
            try:
                analyses.append(self.analyze_ticker(ticker, model_name=model_name, lookback_days=lookback_days))
            except Exception:
                continue

        analyses.sort(key=lambda item: item.predicted_alpha, reverse=True)
        return analyses[:top_n]