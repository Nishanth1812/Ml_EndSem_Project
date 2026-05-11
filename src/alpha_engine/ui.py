from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .engine import DEFAULT_WATCHLIST, FEATURE_COLUMNS, StockAlphaEngine

MODEL_LABELS = {
    "baseline": "Linear Regression Baseline",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_path() -> Path:
    return _project_root() / "data" / "NIFTY 500_day.csv"


def _get_engine() -> StockAlphaEngine:
    if "engine" not in st.session_state:
        st.session_state.engine = StockAlphaEngine(_data_path())
    return st.session_state.engine


def _render_metric_card(title: str, value: str, delta: str | None = None) -> None:
    st.metric(title, value, delta=delta)


def _recommendation(predicted_alpha: float, beta: float) -> tuple[str, str]:
    if predicted_alpha > 0.002 and beta <= 1.2:
        return "STRONG BUY", "#166534"
    if predicted_alpha > 0 and beta <= 1.4:
        return "BUY", "#b45309"
    if predicted_alpha < -0.001:
        return "SELL", "#b91c1c"
    return "HOLD", "#475569"


def run_app() -> None:
    st.set_page_config(page_title="Alpha Intelligence Engine", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 23, 42, 0.08), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }
        h1, h2, h3, h4 {
            color: #0f172a;
            font-family: Inter, Arial, sans-serif;
        }
        .panel {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .formula {
            font-family: Consolas, 'Liberation Mono', monospace;
            background: #0f172a;
            color: #f8fafc;
            padding: 0.8rem 1rem;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    engine = _get_engine()
    st.title("Alpha Intelligence Engine")
    st.caption("Linear Regression baseline with Random Forest and XGBoost as the primary inference models.")

    tabs = st.tabs(["Overview", "Data", "Training", "Inference", "Stock Screener"])

    with tabs[0]:
        st.markdown(
            """
            <div class="panel">
            <h3>Model design</h3>
            <p>The project is now organized around a single supervised return forecast target.</p>
            <p>Linear Regression is the transparent baseline. Random Forest and XGBoost are trained independently and used directly for inference.</p>
            <div class="formula">predicted_alpha = predicted_return - beta x market_expected_return</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Training flow**")
            st.write("1. Load and normalize the price history")
            st.write("2. Create lag, momentum, RSI, volatility, and price-distance features")
            st.write("3. Fit Linear Regression, Random Forest, and XGBoost chronologically")
            st.write("4. Use the selected model for inference and alpha ranking")
        with right:
            concept = go.Figure()
            x_axis = np.linspace(0, 1, 60)
            concept.add_trace(go.Scatter(x=x_axis, y=np.sin(8 * x_axis), name="Market regime", line=dict(color="#94a3b8")))
            concept.add_trace(go.Scatter(x=x_axis, y=np.sin(8 * x_axis) + 0.2 * np.cos(25 * x_axis), name="Model edge", line=dict(color="#0f172a", width=3)))
            concept.update_layout(template="plotly_white", height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(concept, use_container_width=True)

    with tabs[1]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Market data preparation")
        if st.button("Load and prepare data", type="primary"):
            with st.spinner("Preparing price history and features..."):
                try:
                    data = engine.prepare_data()
                    st.success(f"Loaded {len(data):,} cleaned rows with {len(FEATURE_COLUMNS)} model features.")
                except Exception as exc:
                    st.error(str(exc))

        if engine.feature_frame is not None:
            data = engine.feature_frame
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close", line=dict(color="#0f172a", width=2)))
            if "sma_50" in data.columns:
                figure.add_trace(go.Scatter(x=data.index, y=data["sma_50"], name="50-day SMA", line=dict(color="#64748b", dash="dot")))
            figure.update_layout(template="plotly_white", height=360, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(figure, use_container_width=True)
            st.dataframe(data[["close", "rsi_14", "volatility_21", "price_to_sma_50", "target_next_return"]].tail(8), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Train and evaluate models")
        if st.button("Train models", type="primary"):
            with st.spinner("Running chronological training and model tuning..."):
                try:
                    engine.train()
                    st.success("Models trained successfully.")
                except Exception as exc:
                    st.error(str(exc))

        if engine.is_trained:
            metrics_df = engine.metrics_dataframe()
            st.dataframe(metrics_df, use_container_width=True)

            metric_cols = st.columns(3)
            for column, model_name in zip(metric_cols, ["baseline", "random_forest", "xgboost"]):
                metrics = engine.metrics_[model_name]
                with column:
                    _render_metric_card(MODEL_LABELS[model_name], f"MAE {metrics.mae:.6f}", delta=f"R2 {metrics.r2:.4f}")

            holdout = engine.holdout_frame
            if holdout is not None:
                selected_model = st.selectbox(
                    "Holdout view",
                    ["baseline", "random_forest", "xgboost"],
                    format_func=lambda name: MODEL_LABELS[name],
                    key="holdout_model",
                )
                chart = go.Figure()
                chart.add_trace(go.Scatter(x=holdout.index, y=holdout["actual_next_return"], name="Actual", line=dict(color="#0f172a", width=3)))
                chart.add_trace(
                    go.Scatter(
                        x=holdout.index,
                        y=holdout[f"pred_{selected_model}"],
                        name=MODEL_LABELS[selected_model],
                        line=dict(color="#2563eb", width=2),
                    )
                )
                chart.update_layout(template="plotly_white", height=360, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(chart, use_container_width=True)

                importance = engine.feature_importance(selected_model)
                importance_chart = go.Figure(
                    go.Bar(x=importance.head(10).index.tolist(), y=importance.head(10).values.tolist(), marker_color="#0f172a")
                )
                importance_chart.update_layout(template="plotly_white", height=320, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(importance_chart, use_container_width=True)
        else:
            st.info("Train the models to see holdout metrics and feature importance.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Single-stock inference")
        if not engine.is_trained:
            st.info("Train the models first to enable inference.")
        else:
            left, right = st.columns([2, 1])
            with left:
                ticker = st.selectbox("Select a stock", DEFAULT_WATCHLIST, format_func=lambda item: item.replace(".NS", ""))
                model_name = st.selectbox(
                    "Inference model",
                    ["baseline", "random_forest", "xgboost"],
                    index=2,
                    format_func=lambda name: MODEL_LABELS[name],
                )
            with right:
                run_button = st.button("Run inference", type="primary")

            if run_button:
                with st.spinner(f"Scoring {ticker} with {MODEL_LABELS[model_name]}..."):
                    try:
                        analysis = engine.analyze_ticker(ticker, model_name=model_name)
                        metrics = st.columns(4)
                        with metrics[0]:
                            _render_metric_card("Current price", f"₹{analysis.current_price:.2f}")
                        with metrics[1]:
                            _render_metric_card("Predicted return", f"{analysis.predicted_return:.5f}")
                        with metrics[2]:
                            _render_metric_card("Predicted alpha", f"{analysis.predicted_alpha:.5f}")
                        with metrics[3]:
                            _render_metric_card("Beta", f"{analysis.beta:.3f}")

                        tech = st.columns(3)
                        with tech[0]:
                            _render_metric_card("RSI", f"{analysis.rsi:.1f}")
                        with tech[1]:
                            _render_metric_card("Volatility", f"{analysis.volatility:.5f}")
                        with tech[2]:
                            _render_metric_card("Market return", f"{analysis.market_expected_return:.5f}")

                        price_chart = go.Figure()
                        price_chart.add_trace(go.Scatter(x=analysis.data.index, y=analysis.data["close"], name="Close", line=dict(color="#0f172a", width=2)))
                        price_chart.add_trace(go.Scatter(x=analysis.data.index, y=analysis.data["sma_50"], name="50-day SMA", line=dict(color="#94a3b8", dash="dot")))
                        price_chart.update_layout(template="plotly_white", height=360, margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(price_chart, use_container_width=True)

                        recommendation, color = _recommendation(analysis.predicted_alpha, analysis.beta)
                        st.markdown(
                            f"""
                            <div style="background:#ffffff; border-left: 5px solid {color}; padding: 1rem 1.1rem; border-radius: 14px;">
                            <strong>{recommendation}</strong><br/>
                            Predicted alpha: {analysis.predicted_alpha:.5f}<br/>
                            Predicted return: {analysis.predicted_return:.5f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Watchlist ranking")
        if not engine.is_trained:
            st.info("Train the models first to rank the watchlist.")
        else:
            model_name = st.selectbox(
                "Ranking model",
                ["random_forest", "xgboost"],
                index=1,
                format_func=lambda name: MODEL_LABELS[name],
                key="ranking_model",
            )
            top_n = st.slider("Top N stocks", min_value=3, max_value=len(DEFAULT_WATCHLIST), value=5)
            if st.button("Rank watchlist", type="primary"):
                with st.spinner("Scoring the watchlist..."):
                    ranked = engine.rank_tickers(DEFAULT_WATCHLIST, model_name=model_name, top_n=top_n)
                    if ranked:
                        table = pd.DataFrame(
                            [
                                {
                                    "ticker": item.ticker.replace(".NS", ""),
                                    "current_price": item.current_price,
                                    "predicted_return": item.predicted_return,
                                    "predicted_alpha": item.predicted_alpha,
                                    "beta": item.beta,
                                    "volatility": item.volatility,
                                    "rsi": item.rsi,
                                }
                                for item in ranked
                            ]
                        )
                        st.dataframe(table, use_container_width=True)

                        chart = go.Figure(
                            go.Bar(
                                x=table["ticker"],
                                y=table["predicted_alpha"],
                                marker_color="#0f172a",
                            )
                        )
                        chart.update_layout(template="plotly_white", height=320, margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning("No tickers could be scored. Check your internet connection and try again.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()