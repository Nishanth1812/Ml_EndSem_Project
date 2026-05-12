import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import warnings, io, time
warnings.filterwarnings('ignore')

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'rankings' not in st.session_state:
    st.session_state.rankings = None
if 'nifty_cache' not in st.session_state:
    st.session_state.nifty_cache = None
if 'custom_results' not in st.session_state:
    st.session_state.custom_results = None

st.set_page_config(page_title="Alpha Intelligence Engine", layout="wide", page_icon="")

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'alpha_model.pkl'
NIFTY_INDEX = '^NSEI'
NIFTY_100 = [
    'ABB.NS','ADANIENSOL.NS','ADANIENT.NS','ADANIGREEN.NS','ADANIPORTS.NS',
    'ADANIPOWER.NS','AMBUJACEM.NS','APOLLOHOSP.NS','ASIANPAINT.NS','DMART.NS',
    'AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BAJAJHLDNG.NS',
    'BANKBARODA.NS','BEL.NS','BPCL.NS','BHARTIARTL.NS','BOSCHLTD.NS',
    'BRITANNIA.NS','CGPOWER.NS','CANBK.NS','CHOLAFIN.NS','CIPLA.NS',
    'COALINDIA.NS','CUMMINSIND.NS','DLF.NS','DIVISLAB.NS','DRREDDY.NS',
    'EICHERMOT.NS','ETERNAL.NS','GAIL.NS','GODREJCP.NS','GRASIM.NS',
    'HCLTECH.NS','HDFCAMC.NS','HDFCBANK.NS','HDFCLIFE.NS','HINDALCO.NS',
    'HAL.NS','HINDUNILVR.NS','HINDZINC.NS','HYUNDAI.NS','ICICIBANK.NS',
    'ITC.NS','INDHOTEL.NS','IOC.NS','IRFC.NS','INFY.NS',
    'INDIGO.NS','JSWSTEEL.NS','JINDALSTEL.NS','JIOFIN.NS','KOTAKBANK.NS',
    'LTM.NS','LT.NS','LODHA.NS','M&M.NS','MARUTI.NS',
    'MAXHEALTH.NS','MAZDOCK.NS','MUTHOOTFIN.NS','NTPC.NS','NESTLEIND.NS',
    'ONGC.NS','PIDILITIND.NS','PFC.NS','POWERGRID.NS','PNB.NS',
    'RECLTD.NS','RELIANCE.NS','SBILIFE.NS','MOTHERSON.NS','SHREECEM.NS',
    'SHRIRAMFIN.NS','ENRIN.NS','SIEMENS.NS','SOLARINDS.NS','SBIN.NS',
    'SUNPHARMA.NS','TVSMOTOR.NS','TATACAP.NS','TCS.NS','TATACONSUM.NS',
    'TMCV.NS','TMPV.NS','TATAPOWER.NS','TATASTEEL.NS','TECHM.NS',
    'TITAN.NS','TORNTPHARM.NS','TRENT.NS','ULTRACEMCO.NS','UNIONBANK.NS',
    'UNITDSPR.NS','VBL.NS','VEDL.NS','WIPRO.NS','ZYDUSLIFE.NS',
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

models = load_model()
if models is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Run `train_model.py` first.")
    st.stop()

FEATURES = models['feature_cols']

# Dark mode config
def theme():
    if st.session_state.dark_mode:
        return {
            'bg': '#0f1117', 'card': '#1a1d2e', 'border': '#2a2d3e',
            'text': '#e2e8f0', 'muted': '#8892a4',
            'accent': '#3b82f6', 'success': '#22c55e', 'danger': '#ef4444',
            'plotly': 'plotly_dark',
            'header': 'linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%)',
        }
    return {
        'bg': '#f8fafc', 'card': '#ffffff', 'border': '#e2e8f0',
        'text': '#0f172a', 'muted': '#64748b',
        'accent': '#0f172a', 'success': '#16a34a', 'danger': '#dc2626',
        'plotly': 'plotly_white',
        'header': 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
    }

T = theme()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {T['bg']}; color: {T['text']}; }}
.main-header {{ background: {T['header']}; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; border: 1px solid {T['border']}; }}
.main-header h1 {{ color:#fff; font-size:1.8rem; font-weight:800; margin:0; }}
.main-header p {{ color:#94a3b8; font-size:0.95rem; margin:0.3rem 0 0 0; }}
.main-header .badge {{ display:inline-block; background:rgba(255,255,255,0.08); color:#cbd5e1; padding:0.2rem 0.75rem; border-radius:20px; font-size:0.7rem; margin-top:0.5rem; border:1px solid rgba(255,255,255,0.08); }}
.card {{ background:{T['card']}; padding:1.25rem; border-radius:12px; border:1px solid {T['border']}; margin-bottom:1rem; }}
.metric-card {{ background:{T['card']}; padding:1rem; border-radius:12px; border:1px solid {T['border']}; text-align:center; }}
.metric-card .value {{ font-size:1.5rem; font-weight:700; color:{T['text']}; }}
.metric-card .label {{ font-size:0.75rem; color:{T['muted']}; text-transform:uppercase; letter-spacing:0.5px; }}
.explain {{ background:{T['card']}; padding:1.25rem; border-radius:12px; border-left:4px solid {T['accent']}; border-top:1px solid {T['border']}; border-right:1px solid {T['border']}; border-bottom:1px solid {T['border']}; margin-bottom:1rem; color:{T['text']}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:0.5rem; background:{T['card']}; padding:0.4rem; border-radius:12px; border:1px solid {T['border']}; }}
.stTabs [data-baseweb="tab"] {{ border-radius:8px; padding:0.4rem 1rem; font-weight:500; font-size:0.85rem; color:{T['muted']}; }}
.stTabs [aria-selected="true"] {{ background:{T['accent']}!important; color:#fff!important; }}
[data-testid="stMetricValue"] {{ color:{T['text']}!important; }}
section[data-testid="stSidebar"] {{ background:{T['card']}; border-right:1px solid {T['border']}; }}
footer {{ display:none; }}
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("##  Controls")

    model_choice = st.radio(
        "Inference Model",
        ["XGBoost", "Linear Regression"],
        index=0 if models['best_model_name'] == 'XGBoost' else 1,
        help="Select which trained model to use for alpha predictions"
    )
    st.session_state.selected_model = model_choice

    model_key = 'xgboost' if model_choice == 'XGBoost' else 'linear_regression'
    r2_val = models.get('xgb_r2' if model_key == 'xgboost' else 'lr_r2', 0)
    rho_val = models.get('xgb_spearman' if model_key == 'xgboost' else 'lr_spearman', 0)
    mae_val = models.get('xgb_mae' if model_key == 'xgboost' else 'lr_mae', 0)

    st.markdown(f"**{model_choice}**")
    st.markdown(f"R²: `{r2_val:.4f}`  |  Spearman `{rho_val:.4f}`  |  MAE `{mae_val:.6f}`")

    st.markdown("---")
    st.markdown("**Backtest (Long-Short Strategy)**")
    bt = models.get('backtest', {})
    if bt:
        sel = bt.get('best', {})
        st.markdown(f"Sharpe: `{sel.get('sharpe', 0):.2f}`  |  Win: `{sel.get('win_rate', 0):.0%}`")
        st.markdown(f"Max DD: `{sel.get('max_drawdown', 0):.4f}`  |  Avg Spread: `{sel.get('avg_spread', 0):.4f}`")

    st.markdown("---")
    st.markdown(f"**Model Card**")
    st.markdown(f"Trained: {models.get('training_date', 'N/A')}")
    st.markdown(f"Data: {models.get('n_stocks', 0)} stocks, {models.get('n_training_samples', 0):,} rows")
    st.markdown(f"Target: 21-day forward alpha")

    st.markdown("---")
    dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

# ========== HEADER ==========
st.markdown(f"""
<div class="main-header">
<h1></h1>
<p>Alpha-Driven Portfolio Recommendation System</p>
<span class="badge"> Nifty 100 ({models.get('n_stocks',0)} stocks)  •  LR + XGBoost  •  21-Day Alpha Target  •  Sharpe {bt.get('best',{}).get('sharpe',0):.2f} Backtest</span>
</div>
""", unsafe_allow_html=True)

# ========== MODEL INFERENCE HELPERS ==========
def compute_features(df):
    out = df.copy()
    out['Returns'] = np.log(out['Close'] / out['Close'].shift(1))
    out['SMA_50'] = out['Close'].rolling(50).mean()
    out['SMA_20'] = out['Close'].rolling(20).mean()
    out['Price_to_SMA_50'] = out['Close'] / out['SMA_50']
    out['Price_to_SMA_20'] = out['Close'] / out['SMA_20']
    out['SMA_Crossover'] = out['SMA_20'] / out['SMA_50']
    delta = out['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    out['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    out['Volatility'] = out['Returns'].rolling(21).std()
    for lag in [1, 2, 3, 5]:
        out[f'Return_Lag_{lag}'] = out['Returns'].shift(lag)
    out['Return_Momentum_5'] = out['Returns'].rolling(5).mean()
    out['Return_Momentum_10'] = out['Returns'].rolling(10).mean()
    out['Return_Momentum_21'] = out['Returns'].rolling(21).mean()
    return out

def predict_alpha_for_all(tickers, progress_callback=None):
    """Fetch fresh data for all stocks and predict alpha using selected model."""
    model_key = 'xgboost' if st.session_state.selected_model == 'XGBoost' else 'linear_regression'
    infer_model = models[model_key]
    scaler = models['scaler']

    end = datetime.now()
    start = end - timedelta(days=730)

    # Fetch NIFTY
    if st.session_state.nifty_cache is not None:
        nifty = st.session_state.nifty_cache
    else:
        nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty = nifty.droplevel(1, axis=1)
        st.session_state.nifty_cache = nifty

    nifty['Market_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))

    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        time.sleep(0.15)  # avoid yfinance rate limiting
        try:
            d = yf.download(ticker, start=start, end=end, progress=False)
            if d.empty or isinstance(d.columns, pd.MultiIndex) and d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1)
            if 'Close' not in d.columns or len(d) < 200:
                continue

            df = compute_features(d)
            df = df.join(nifty[['Market_Return']], how='inner')
            df['Market_Return_Lag_1'] = df['Market_Return'].shift(1)

            cov = df['Returns'].rolling(60).cov(df['Market_Return'])
            var = df['Market_Return'].rolling(60).var() + 1e-9
            df['Rolling_Beta'] = (cov / var).clip(-2, 5)
            df['Alpha'] = df['Returns'] - df['Rolling_Beta'] * df['Market_Return']

            df['Excess_Return_5'] = df['Returns'].rolling(5).mean() - df['Market_Return'].rolling(5).mean()
            df['Excess_Return_10'] = df['Returns'].rolling(10).mean() - df['Market_Return'].rolling(10).mean()
            df['Excess_Return_21'] = df['Returns'].rolling(21).mean() - df['Market_Return'].rolling(21).mean()
            df['52w_High'] = df['Close'].rolling(252).max()
            df['Price_vs_52w_High'] = df['Close'] / df['52w_High']
            df['Day_of_Week'] = df.index.dayofweek
            df['Month_of_Year'] = df.index.month

            latest = df.dropna(subset=FEATURES).iloc[-1:]
            if latest.empty:
                continue

            feat = latest[FEATURES].values
            feat_s = scaler.transform(feat)
            pred = infer_model.predict(feat_s)[0]

            results.append({
                'ticker': ticker.replace('.NS', ''),
                'predicted_alpha': float(pred),
                'beta': float(latest['Rolling_Beta'].iloc[0]),
                'volatility': float(latest['Volatility'].iloc[0]),
                'rsi': float(latest['RSI'].iloc[0]),
                'price': float(latest['Close'].iloc[0]),
            })
        except:
            pass

        if progress_callback:
            progress_callback((i + 1) / total)

    return sorted(results, key=lambda x: x['predicted_alpha'], reverse=True)

def predict_alpha_for_custom_tickers(tickers):
    """Predict alpha for arbitrary user-entered tickers (not limited to Nifty 100)."""
    model_key = 'xgboost' if st.session_state.selected_model == 'XGBoost' else 'linear_regression'
    infer_model = models[model_key]
    scaler = models['scaler']

    end = datetime.now()
    start = end - timedelta(days=730)

    if st.session_state.nifty_cache is not None:
        nifty = st.session_state.nifty_cache
    else:
        nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty = nifty.droplevel(1, axis=1)
        st.session_state.nifty_cache = nifty

    nifty['Market_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))

    results = []
    errors = []
    for ticker in tickers:
        time.sleep(0.15)
        try:
            full = ticker if ticker.endswith('.NS') else ticker + '.NS'
            d = yf.download(full, start=start, end=end, progress=False)
            if d.empty:
                errors.append((ticker, 'No data from Yahoo Finance'))
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1)
            if 'Close' not in d.columns or len(d) < 200:
                errors.append((ticker, 'Insufficient history (<200 days)'))
                continue

            df = compute_features(d)
            df = df.join(nifty[['Market_Return']], how='inner')
            df['Market_Return_Lag_1'] = df['Market_Return'].shift(1)

            cov = df['Returns'].rolling(60).cov(df['Market_Return'])
            var = df['Market_Return'].rolling(60).var() + 1e-9
            df['Rolling_Beta'] = (cov / var).clip(-2, 5)
            df['Alpha_hist'] = df['Returns'] - df['Rolling_Beta'] * df['Market_Return']

            df['Excess_Return_5'] = df['Returns'].rolling(5).mean() - df['Market_Return'].rolling(5).mean()
            df['Excess_Return_10'] = df['Returns'].rolling(10).mean() - df['Market_Return'].rolling(10).mean()
            df['Excess_Return_21'] = df['Returns'].rolling(21).mean() - df['Market_Return'].rolling(21).mean()
            df['52w_High'] = df['Close'].rolling(252).max()
            df['Price_vs_52w_High'] = df['Close'] / df['52w_High']
            df['Day_of_Week'] = df.index.dayofweek
            df['Month_of_Year'] = df.index.month

            latest = df.dropna(subset=FEATURES).iloc[-1:]
            if latest.empty:
                errors.append((ticker, 'Insufficient feature data'))
                continue

            feat = latest[FEATURES].values
            feat_s = scaler.transform(feat)
            pred = infer_model.predict(feat_s)[0]

            results.append({
                'ticker': ticker.upper().replace('.NS', ''),
                'predicted_alpha': float(pred),
                'beta': float(latest['Rolling_Beta'].iloc[0]),
                'volatility': float(latest['Volatility'].iloc[0]),
                'rsi': float(latest['RSI'].iloc[0]),
                'price': float(latest['Close'].iloc[0]),
            })
        except Exception as e:
            errors.append((ticker, str(e)[:60]))

    return sorted(results, key=lambda x: x['predicted_alpha'], reverse=True), errors

# ========== TABS ==========
tabs = st.tabs([" Framework", " Backtest", " Alpha Rankings", " Portfolio", " Stock Analysis", " Custom Portfolio"])

# ========== TAB 0: FRAMEWORK ==========
with tabs[0]:
    st.markdown(f"""
    <div class="explain">
    <h3>What is Alpha?</h3>
    <p>Alpha = Stock Return − (Beta × Market Return). The model predicts <strong>21-day forward cumulative alpha</strong> — the excess return over beta-adjusted market performance over the next month.</p>
    <p style="font-size:0.85rem;color:{T['muted']};">Why 21 days? Daily alpha is dominated by noise. A monthly horizon captures meaningful stock-specific signals while being a realistic rebalancing period for portfolio managers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Architecture")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card"><h3>1. Data</h3><p>{models.get('n_stocks',0)} Nifty stocks from yfinance. 20 features: technicals, momentum (5/10/21d), excess returns, 52-week high, rolling beta, seasonality.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><h3>2. Models</h3><p><b>Linear Regression:</b> R² {models.get('lr_r2',0):.4f}, Spearman {models.get('lr_spearman',0):.4f}<br><b>XGBoost:</b> R² {models.get('xgb_r2',0):.4f}, Spearman {models.get('xgb_spearman',0):.4f}</p></div>""", unsafe_allow_html=True)
    with c3:
        bt_data = models.get('backtest', {}).get('best', {})
        st.markdown(f"""<div class="card"><h3>3. Strategy</h3><p>Long top 20%, short bottom 20%. Monthly rebalance.<br><b>Sharpe: {bt_data.get('sharpe',0):.2f}</b> | Win: {bt_data.get('win_rate',0):.0%} | Max DD: {bt_data.get('max_drawdown',0):.3f}</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card"><h3>4. Portfolio</h3><p>Mean-variance optimization on top-alpha stocks. Long-only, max 10% per stock. Efficient frontier shows risk-return tradeoff.</p></div>""", unsafe_allow_html=True)

    st.subheader("Feature List")
    feat_df = pd.DataFrame({'Feature': FEATURES, 'Description': [
        'Relative Strength Index (14-day)', '21-day rolling std of returns',
        'Close / 50-day SMA', 'Close / 20-day SMA', '20-day SMA / 50-day SMA',
        '1-day lagged return', '2-day lagged return', '3-day lagged return', '5-day lagged return',
        '5-day momentum', '10-day momentum', '21-day momentum',
        '60-day rolling beta vs NIFTY', '1-day lagged market return',
        '5-day excess return vs market', '10-day excess return vs market', '21-day excess return vs market',
        'Close / 52-week high', 'Day of week', 'Month of year',
    ]})
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ========== TAB 1: BACKTEST ==========
with tabs[1]:
    st.subheader("Walk-Forward Backtest")

    bt = models.get('backtest', {})
    if not bt:
        st.info("No backtest data available.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        best_bt = bt.get('best', {})
        col1.metric("Sharpe Ratio (Ann.)", f"{best_bt.get('sharpe', 0):.2f}")
        col2.metric("Win Rate", f"{best_bt.get('win_rate', 0):.0%}")
        col3.metric("Max Drawdown", f"{best_bt.get('max_drawdown', 0):.4f}")
        col4.metric("Avg Monthly Spread", f"{best_bt.get('avg_spread', 0):.4f}")

        st.markdown("### Equity Curve (Long-Short Strategy)")
        cum = best_bt.get('cumulative', [])
        if cum:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=cum, name='Strategy', line=dict(color=T['accent'], width=2),
                                      fill='tozeroy', fillcolor=f'rgba(59,130,246,0.1)'))
            fig.update_layout(template=T['plotly'], xaxis_title="Batch", yaxis_title="Cumulative Alpha Spread",
                              height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Spread by Batch")
        spreads = best_bt.get('spreads', [])
        if spreads:
            colors = [T['success'] if s > 0 else T['danger'] for s in spreads]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=list(range(len(spreads))), y=spreads, marker_color=colors))
            fig2.update_layout(template=T['plotly'], xaxis_title="Batch", yaxis_title="Alpha Spread",
                               height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Model Comparison")
        comp = pd.DataFrame({
            'Metric': ['R²', 'Spearman ρ', 'Sharpe (ann.)', 'Win Rate', 'Max DD'],
            'Linear Regression': [
                f"{models.get('lr_r2', 0):.4f}",
                f"{models.get('lr_spearman', 0):.4f}",
                f"{bt.get('lr', {}).get('sharpe', 0):.2f}",
                f"{bt.get('lr', {}).get('win_rate', 0):.0%}",
                f"{bt.get('lr', {}).get('max_drawdown', 0):.4f}",
            ],
            'XGBoost': [
                f"{models.get('xgb_r2', 0):.4f}",
                f"{models.get('xgb_spearman', 0):.4f}",
                f"{bt.get('xgb', {}).get('sharpe', 0):.2f}",
                f"{bt.get('xgb', {}).get('win_rate', 0):.0%}",
                f"{bt.get('xgb', {}).get('max_drawdown', 0):.4f}",
            ],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="explain">
        <b>Note:</b> R² near zero is expected for alpha prediction — alpha is the residual after removing market beta.
        The model's value is in <b>ranking separation</b>. The current model achieves Sharpe {bt.get('best',{}).get('sharpe',0):.2f}
        with {bt.get('best',{}).get('win_rate',0):.0%} win rate — the long-short portfolio consistently extracts positive alpha.
        </div>
        """, unsafe_allow_html=True)

# ========== TAB 2: ALPHA RANKINGS ==========
with tabs[2]:
    st.subheader("Alpha Rankings")

    if st.button(" Fetch & Rank All Stocks", type="primary"):
        with st.spinner("Fetching NIFTY index..."):
            end = datetime.now()
            start = end - timedelta(days=730)
            nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty = nifty.droplevel(1, axis=1)
            st.session_state.nifty_cache = nifty

        with st.spinner(f"Analyzing {len(NIFTY_100)} stocks..."):
            pb = st.progress(0)
            def upd(p):
                pb.progress(p)
            rankings = predict_alpha_for_all(NIFTY_100, progress_callback=upd)

        if rankings:
            st.session_state.rankings = rankings
            st.success(f"Ranked {len(rankings)} stocks using **{st.session_state.selected_model}**")

    if st.session_state.rankings:
        rankings = st.session_state.rankings
        df = pd.DataFrame(rankings)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        df['Predicted Alpha'] = df['predicted_alpha'].round(6)
        df['Price'] = df['price'].round(2)
        df['Beta'] = df['beta'].round(3)
        df['Volatility'] = df['volatility'].round(4)
        df['RSI'] = df['rsi'].round(1)

        st.dataframe(df[['Rank', 'ticker', 'Predicted Alpha', 'Price', 'Beta', 'Volatility', 'RSI']],
                     use_container_width=True, hide_index=True,
                     column_config={
                         'Predicted Alpha': st.column_config.NumberColumn(format="%.6f"),
                         'Price': st.column_config.NumberColumn(format="₹%.2f"),
                     })

        top20 = df.head(20)
        colors = [T['success'] if a > 0 else T['danger'] for a in top20['predicted_alpha']]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top20['ticker'], y=top20['predicted_alpha'],
                              marker_color=colors, text=top20['predicted_alpha'].round(4),
                              textposition='outside'))
        fig.update_layout(title="Top 20 Stocks by Predicted Alpha", xaxis_title="Stock",
                          yaxis_title="Predicted Alpha", template=T['plotly'], height=400,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(" Download Rankings", data=csv_buf.getvalue(),
                           file_name="alpha_rankings.csv", mime="text/csv")
    else:
        st.info("Click **Fetch & Rank All Stocks** to generate predictions.")

# ========== TAB 3: PORTFOLIO ==========
with tabs[3]:
    st.subheader("Portfolio Optimization")

    if not st.session_state.rankings:
        st.info("Rank stocks first in the **Alpha Rankings** tab.")
    else:
        with st.expander("Optimization Settings", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                n_stocks = st.slider("Number of stocks in portfolio", 5, 20, 10)
                risk_free = st.number_input("Risk-free rate (annual %)", value=6.5, step=0.5) / 100
            with col_s2:
                max_weight = st.slider("Max weight per stock (%)", 10, 40, 20) / 100
                target_vol = st.slider("Target annualized volatility (%)", 5, 30, 15) / 100

        if st.button(" Optimize Portfolio", type="primary"):
            rankings = st.session_state.rankings[:n_stocks]

            alphas = np.array([s['predicted_alpha'] for s in rankings])
            vols = np.array([s['volatility'] for s in rankings])
            betas = np.array([s['beta'] for s in rankings])
            tickers = [s['ticker'] for s in rankings]

            # Assume uniform correlation for simplicity
            corr = 0.3
            cov_matrix = np.outer(vols, vols) * corr
            np.fill_diagonal(cov_matrix, vols ** 2)
            cov_matrix *= 21  # Scale to monthly (21 trading days)

            n = len(tickers)
            monthly_rf = risk_free / 12

            # Efficient frontier via Monte Carlo
            n_portfolios = 5000
            np.random.seed(42)
            all_weights = np.random.dirichlet(np.ones(n), n_portfolios)
            # Apply max weight constraint
            for w in all_weights:
                if w.max() > max_weight:
                    w[:] = np.clip(w, 0, max_weight)
                    w /= w.sum()

            port_returns = all_weights @ alphas
            port_vols = np.sqrt(np.diag(all_weights @ cov_matrix @ all_weights.T))
            port_sharpes = (port_returns - monthly_rf) / (port_vols + 1e-9)

            # Max Sharpe portfolio
            best_idx = np.argmax(port_sharpes)
            best_w = all_weights[best_idx]
            best_ret = port_returns[best_idx]
            best_vol = port_vols[best_idx]
            best_sharpe = port_sharpes[best_idx]

            # Min Vol portfolio
            min_idx = np.argmin(port_vols)
            min_w = all_weights[min_idx]
            min_ret = port_returns[min_idx]
            min_vol = port_vols[min_idx]

            st.markdown("### Optimized Portfolio (Max Sharpe)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Monthly Alpha", f"{best_ret:.4f}")
            m2.metric("Monthly Volatility", f"{best_vol:.4f}")
            m3.metric("Sharpe Ratio", f"{best_sharpe:.2f}")
            m4.metric("Annualized Alpha", f"{best_ret * 12:.4f}")

            port_df = pd.DataFrame({
                'Stock': tickers,
                'Predicted Alpha': [f"{a:.6f}" for a in alphas],
                'Beta': [f"{b:.3f}" for b in betas],
                'Volatility': [f"{v:.4f}" for v in vols],
                'Weight': [f"{w*100:.2f}%" for w in best_w],
            })
            st.dataframe(port_df, use_container_width=True, hide_index=True)

            # Efficient Frontier
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=port_vols, y=port_returns, mode='markers',
                                         marker=dict(color='#94a3b8', size=3, opacity=0.4),
                                         name='Random Portfolios'))
            fig_ef.add_trace(go.Scatter(x=[best_vol], y=[best_ret], mode='markers',
                                         marker=dict(color=T['success'], size=14, symbol='star'),
                                         name='Max Sharpe'))
            fig_ef.add_trace(go.Scatter(x=[min_vol], y=[min_ret], mode='markers',
                                         marker=dict(color=T['accent'], size=12, symbol='diamond'),
                                         name='Min Vol'))
            fig_ef.update_layout(title="Efficient Frontier", xaxis_title="Monthly Volatility",
                                  yaxis_title="Expected Monthly Alpha", template=T['plotly'], height=400,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ef, use_container_width=True)

            # Allocation Pie
            fig_pie = go.Figure(data=[go.Pie(labels=tickers, values=best_w, hole=0.4,
                                              marker=dict(colors=px.colors.qualitative.Bold))])
            fig_pie.update_layout(title="Portfolio Allocation", template=T['plotly'], height=400,
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)

            csv_b = io.StringIO()
            port_df.to_csv(csv_b, index=False)
            st.download_button(" Download Portfolio", data=csv_b.getvalue(),
                               file_name="optimized_portfolio.csv", mime="text/csv")

# ========== TAB 4: STOCK ANALYSIS ==========
with tabs[4]:
    st.subheader("Individual Stock Analysis")

    ticker_input = st.text_input("Enter NSE stock symbol (e.g. RELIANCE, TCS, HDFCBANK)", "RELIANCE").strip().upper()
    full_ticker = ticker_input + '.NS'

    if st.button(" Analyze", type="primary"):
        with st.spinner("Fetching data..."):
            model_key = 'xgboost' if st.session_state.selected_model == 'XGBoost' else 'linear_regression'
            infer_model = models[model_key]
            scaler = models['scaler']

            end = datetime.now()
            start = end - timedelta(days=730)

            if st.session_state.nifty_cache is not None:
                nifty = st.session_state.nifty_cache
            else:
                nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
                if isinstance(nifty.columns, pd.MultiIndex):
                    nifty = nifty.droplevel(1, axis=1)
                st.session_state.nifty_cache = nifty

            nifty['Market_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))

            try:
                d = yf.download(full_ticker, start=start, end=end, progress=False)
                if d.empty:
                    st.error(f"No data for {ticker_input}")
                    st.stop()
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.droplevel(1, axis=1)
                if 'Close' not in d.columns:
                    st.error("Missing Close price data.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed: {e}")
                st.stop()

            df = compute_features(d)
            df = df.join(nifty[['Market_Return']], how='inner')
            df['Market_Return_Lag_1'] = df['Market_Return'].shift(1)

            cov = df['Returns'].rolling(60).cov(df['Market_Return'])
            var = df['Market_Return'].rolling(60).var() + 1e-9
            df['Rolling_Beta'] = (cov / var).clip(-2, 5)
            df['Alpha'] = df['Returns'] - df['Rolling_Beta'] * df['Market_Return']

            df['Excess_Return_5'] = df['Returns'].rolling(5).mean() - df['Market_Return'].rolling(5).mean()
            df['Excess_Return_10'] = df['Returns'].rolling(10).mean() - df['Market_Return'].rolling(10).mean()
            df['Excess_Return_21'] = df['Returns'].rolling(21).mean() - df['Market_Return'].rolling(21).mean()
            df['52w_High'] = df['Close'].rolling(252).max()
            df['Price_vs_52w_High'] = df['Close'] / df['52w_High']
            df['Day_of_Week'] = df.index.dayofweek
            df['Month_of_Year'] = df.index.month

            latest = df.dropna(subset=FEATURES).iloc[-1:]
            if latest.empty:
                st.error("Insufficient data for prediction.")
                st.stop()

            feat = latest[FEATURES].values
            feat_s = scaler.transform(feat)
            pred = infer_model.predict(feat_s)[0]

            st.markdown(f"### {ticker_input} — Key Metrics")
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Current Price", f"{latest['Close'].iloc[0]:.2f}")
            a2.metric(f"Predicted Alpha (21d)", f"{float(pred):.6f}")
            a3.metric("Rolling Beta", f"{latest['Rolling_Beta'].iloc[0]:.3f}")
            a4.metric("Volatility", f"{latest['Volatility'].iloc[0]:.4f}")
            a5.metric("RSI", f"{latest['RSI'].iloc[0]:.1f}")

            st.markdown("### Historical Alpha")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Cumulative Monthly Alpha", "Daily Alpha"))
            mc = df['Alpha'].rolling(21).sum().dropna()
            fig.add_trace(go.Scatter(x=mc.index, y=mc.cumsum(), name="Cumulative Alpha",
                                      line=dict(color=T['accent'])), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Alpha'], name="Daily Alpha",
                                      line=dict(color='#94a3b8', width=1)), row=2, col=1)
            fig.update_layout(template=T['plotly'], height=500, showlegend=False,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Price Chart")
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color=T['accent'])))
            fig_p.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-day SMA',
                                        line=dict(color='#cbd5e1', dash='dot')))
            fig_p.update_layout(template=T['plotly'], xaxis_title="Date", yaxis_title="Price",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_p, use_container_width=True)

# ========== TAB 5: CUSTOM PORTFOLIO ==========
with tabs[5]:
    st.subheader("Custom Portfolio Builder")
    st.markdown("""Enter any NSE stocks — the model will fetch their data, predict 21-day alpha, 
and let you assign weights manually to build a custom portfolio.""")

    ticker_input = st.text_area(
        "Stock symbols (comma or newline separated)",
        "RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK",
        height=90,
        help="e.g. RELIANCE, TCS, SBIN or one per line"
    )

    col_fetch, col_status = st.columns([1, 3])
    with col_fetch:
        fetch_clicked = st.button(" Fetch & Analyze", type="primary")
    with col_status:
        status_placeholder = st.empty()

    if fetch_clicked:
        raw = ticker_input.replace(',', '\n')
        tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
        if not tickers:
            st.warning("Enter at least one stock symbol.")
        else:
            with st.spinner(f"Analyzing {len(tickers)} stocks (may take a minute)..."):
                results, errors = predict_alpha_for_custom_tickers(tickers)

            if results:
                st.session_state.custom_results = results
                status_placeholder.success(f"Analyzed {len(results)}/{len(tickers)} stocks")
            if errors:
                err_msgs = [f"**{sym}**: {err}" for sym, err in errors]
                st.warning("Failed to analyze:\n" + "\n".join(err_msgs))

    if st.session_state.custom_results:
        results = st.session_state.custom_results
        df = pd.DataFrame(results)
        n = len(results)

        st.markdown("### Analysis Results")
        display = df[['ticker', 'predicted_alpha', 'price', 'beta', 'volatility', 'rsi']].copy()
        display.columns = ['Ticker', 'Predicted Alpha', 'Price', 'Beta', 'Vol.', 'RSI']
        display['Predicted Alpha'] = display['Predicted Alpha'].round(6)
        display['Price'] = display['Price'].round(2)
        display['Beta'] = display['Beta'].round(3)
        display['Vol.'] = display['Vol.'].round(4)
        display['RSI'] = display['RSI'].round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)

        fig_bar = go.Figure()
        colors = [T['success'] if a > 0 else T['danger'] for a in df['predicted_alpha']]
        fig_bar.add_trace(go.Bar(x=df['ticker'], y=df['predicted_alpha'],
                                  marker_color=colors,
                                  text=df['predicted_alpha'].round(4),
                                  textposition='outside'))
        fig_bar.update_layout(title="Predicted Alpha by Stock", template=T['plotly'],
                               height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Manual Weight Assignment")
        st.markdown("Adjust each stock's weight below. Weights are normalized to sum to 100%.")

        cols = st.columns(min(n, 5))
        weight_inputs = {}
        equal_w = round(100.0 / n, 1)
        for i, r in enumerate(results):
            ci = i % len(cols)
            with cols[ci]:
                weight_inputs[r['ticker']] = st.number_input(
                    f"{r['ticker']} %", min_value=0.0, max_value=100.0,
                    value=equal_w, step=1.0, key=f"cw_{i}"
                )

        if st.button(" Compute Portfolio Metrics", type="primary"):
            raw_weights = np.array([weight_inputs[r['ticker']] for r in results])
            total_raw = raw_weights.sum()
            if total_raw <= 0:
                st.error("Total weight must be > 0.")
            else:
                w = raw_weights / total_raw

                weighted_alpha = float(w @ df['predicted_alpha'].values)
                weighted_beta = float(w @ df['beta'].values)

                vols = df['volatility'].values
                corr = 0.3
                cov_m = np.outer(vols, vols) * corr
                np.fill_diagonal(cov_m, vols ** 2)
                cov_m *= 21
                port_vol = float(np.sqrt(w @ cov_m @ w))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Monthly Alpha", f"{weighted_alpha:.6f}")
                m2.metric("Portfolio Beta", f"{weighted_beta:.3f}")
                m3.metric("Monthly Volatility", f"{port_vol:.4f}")
                m4.metric("Annualized Alpha", f"{weighted_alpha * 12:.4f}")

                st.markdown("### Allocation")
                alloc = pd.DataFrame({
                    'Stock': [r['ticker'] for r in results],
                    'Predicted Alpha': [f"{r['predicted_alpha']:.6f}" for r in results],
                    'Beta': [f"{r['beta']:.3f}" for r in results],
                    'Weight': [f"{wi*100:.2f}%" for wi in w],
                })
                st.dataframe(alloc, use_container_width=True, hide_index=True)

                fig_pie = go.Figure(data=[go.Pie(
                    labels=[r['ticker'] for r in results],
                    values=w * 100, hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Bold)
                )])
                fig_pie.update_layout(title="Portfolio Allocation", template=T['plotly'], height=400,
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                st.plotly_chart(fig_pie, use_container_width=True)

                csv_b = io.StringIO()
                alloc.to_csv(csv_b, index=False)
                st.download_button(" Download Portfolio", data=csv_b.getvalue(),
                                   file_name="custom_portfolio.csv", mime="text/csv")
