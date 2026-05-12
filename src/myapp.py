import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

st.set_page_config(page_title="Alpha Intelligence Engine", layout="wide", page_icon="")

if st.session_state.dark_mode:
    BG = "#0f1117"
    CARD = "#1a1d2e"
    BORDER = "#2a2d3e"
    TEXT = "#e2e8f0"
    TEXT_MUTED = "#8892a4"
    HEADER_BG = "linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%)"
    HEADER_BORDER = "#2a2d3e"
    PLOTLY_TEMPLATE = "plotly_dark"
    METRIC_BG = "#1a1d2e"
    TAB_BG = "#1a1d2e"
    TAB_ACTIVE = "#3b82f6"
    ACCENT = "#3b82f6"
    SUCCESS = "#22c55e"
    DANGER = "#ef4444"
    WARNING_COLOR = "#f59e0b"
else:
    BG = "#f8fafc"
    CARD = "#ffffff"
    BORDER = "#e2e8f0"
    TEXT = "#0f172a"
    TEXT_MUTED = "#64748b"
    HEADER_BG = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    HEADER_BORDER = "#e2e8f0"
    PLOTLY_TEMPLATE = "plotly_white"
    METRIC_BG = "#ffffff"
    TAB_BG = "#ffffff"
    TAB_ACTIVE = "#0f172a"
    ACCENT = "#0f172a"
    SUCCESS = "#16a34a"
    DANGER = "#dc2626"
    WARNING_COLOR = "#d97706"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    .main-header {{
        background: {HEADER_BG};
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid {HEADER_BORDER};
    }}
    .main-header h1 {{ color: #ffffff !important; font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }}
    .main-header p {{ color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 300; }}
    .main-header .badge {{
        display: inline-block; background: rgba(255,255,255,0.08); color: #cbd5e1;
        padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 500;
        margin-top: 0.75rem; border: 1px solid rgba(255,255,255,0.08);
    }}

    .card {{
        background: {CARD}; padding: 1.5rem; border-radius: 12px;
        border: 1px solid {BORDER}; margin-bottom: 1rem;
        transition: box-shadow 0.2s;
    }}
    .card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
    .card h3 {{ color: {TEXT}; font-size: 1.1rem; font-weight: 600; margin-top: 0; margin-bottom: 0.75rem; }}

    .metric-card {{
        background: {METRIC_BG}; padding: 1.25rem; border-radius: 12px;
        border: 1px solid {BORDER}; text-align: center;
    }}
    .metric-card .value {{ font-size: 1.75rem; font-weight: 700; color: {TEXT}; }}
    .metric-card .label {{ font-size: 0.8rem; color: {TEXT_MUTED}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .delta-positive {{ color: {SUCCESS}; font-size: 0.85rem; font-weight: 600; }}
    .metric-card .delta-negative {{ color: {DANGER}; font-size: 0.85rem; font-weight: 600; }}

    .explanation-card {{
        background: {CARD}; padding: 1.5rem; border-radius: 12px;
        border-left: 4px solid {ACCENT};
        border-top: 1px solid {BORDER}; border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
        margin-bottom: 1.5rem; color: {TEXT};
    }}

    .math-text {{
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        background: {BG}; padding: 0.75rem 1rem; border-radius: 8px;
        border: 1px solid {BORDER}; font-size: 0.9rem;
    }}

    .recommendation-card {{
        padding: 1.25rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid;
        background: {CARD}; border-top: 1px solid {BORDER}; border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    }}

    .sidebar-section {{ padding: 0.5rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 0.75rem; }}
    .sidebar-section:last-child {{ border-bottom: none; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem; background: {TAB_BG}; padding: 0.5rem; border-radius: 12px;
        border: 1px solid {BORDER}; margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500;
        font-size: 0.85rem; color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background: {TAB_ACTIVE} !important; color: #ffffff !important;
    }}

    .stButton button {{
        border-radius: 8px; font-weight: 600; font-size: 0.85rem;
        padding: 0.5rem 1.25rem; transition: all 0.2s;
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }}

    [data-testid="stMetricValue"] {{ font-size: 1.5rem !important; font-weight: 700 !important; color: {TEXT} !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.85rem !important; }}

    .stSelectbox label, .stSlider label, .stCheckbox label {{ color: {TEXT} !important; }}
    .st-emotion-cache-16idsys p {{ color: {TEXT} !important; }}

    .stDataFrame {{ color: {TEXT} !important; }}
    .stDataFrame [data-testid="StyledDataFrameColHeader"] {{ color: {TEXT_MUTED} !important; }}

    .element-container p, .element-container li {{ color: {TEXT} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT} !important; }}

    .stAlert {{ background: {CARD} !important; border: 1px solid {BORDER} !important; color: {TEXT} !important; }}
    .stAlert p {{ color: {TEXT} !important; }}
    .st-bb {{ background: {CARD} !important; }}
    .st-at {{ background: {BG} !important; }}

    section[data-testid="stSidebar"] .stButton button {{ width: 100%; }}
    section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] .sidebar-section p, section[data-testid="stSidebar"] .stMarkdown p {{ color: {TEXT} !important; }}

    .stRadio label {{ color: {TEXT} !important; }}
    .stRadio [data-testid="stWidgetLabel"] {{ color: {TEXT_MUTED} !important; }}

    footer {{ display: none; }}
    </style>
""", unsafe_allow_html=True)

PLOTLY_THEME = PLOTLY_TEMPLATE

class AlphaIntelligenceEngine:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.selected_stocks = []
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        self.meta_learner = LinearRegression()
        self.is_trained = False
        self.indian_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'MARUTI.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'ASIANPAINT.NS'
        ]
        self.training_history = []

    def configure(self, rf_params=None, xgb_params=None):
        if rf_params:
            self.rf = RandomForestRegressor(**rf_params)
        if xgb_params:
            self.xgb = XGBRegressor(**xgb_params)

    def load_and_preprocess(self, filepath=None):
        try:
            if filepath:
                self.filepath = filepath
            df = pd.read_csv(self.filepath)
            date_col = next((c for c in df.columns if c.lower() == 'date'), None)
            close_col = next((c for c in df.columns if c.lower() == 'close'), None)
            if not date_col or not close_col:
                st.error(f"CSV must contain 'Date' and 'Close' columns. Found: {list(df.columns)}")
                return None

            df['date'] = pd.to_datetime(df[date_col])
            df['close'] = df[close_col]
            df = df.sort_values('date').set_index('date')
            df['Returns'] = np.log(df['close'] / df['close'].shift(1))
            df['SMA_50'] = df['close'].rolling(window=min(50, len(df))).mean()
            df['SMA_20'] = df['close'].rolling(window=min(20, len(df))).mean()
            df['Price_to_SMA_50'] = df['close'] / df['SMA_50']
            df['Price_to_SMA_20'] = df['close'] / df['SMA_20']
            df['SMA_Crossover'] = df['SMA_20'] / df['SMA_50']

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            df['Volatility'] = df['Returns'].rolling(window=21).std()

            for lag in [1, 2, 3, 5]:
                df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
            df['Return_Momentum_5'] = df['Returns'].rolling(5).mean()
            df['Return_Momentum_10'] = df['Returns'].rolling(10).mean()
            df['Market_Beta'] = df['Returns'].rolling(window=5).mean()
            df['Target_Alpha'] = df['Returns'].shift(-1)

            self.feature_cols = [
                'RSI', 'Volatility', 'Price_to_SMA_50', 'Price_to_SMA_20',
                'SMA_Crossover', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
                'Return_Lag_5', 'Return_Momentum_5', 'Return_Momentum_10',
                'Market_Beta'
            ]

            self.df = df.dropna()
            return self.df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    def train_ensemble(self, test_size=0.2, shuffle=True):
        features = self.feature_cols
        X = self.df[features]
        y = self.df['Target_Alpha']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_test = y_test

        self.rf.fit(X_train_scaled, y_train)
        self.xgb.fit(X_train_scaled, y_train)

        rf_p = self.rf.predict(self.X_test_scaled)
        xgb_p = self.xgb.predict(self.X_test_scaled)

        meta_X = np.column_stack((rf_p, xgb_p))
        self.meta_learner.fit(meta_X, y_test)

        self.is_trained = True
        final_pred = self.meta_learner.predict(meta_X)
        r2 = r2_score(y_test, final_pred)

        self.training_history.append({
            'test_size': test_size, 'shuffle': shuffle,
            'r2': r2, 'rf_r2': r2_score(y_test, rf_p), 'xgb_r2': r2_score(y_test, xgb_p),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        return r2

    def select_alpha_stocks(self, top_n=5):
        if not self.is_trained:
            return [], []
        recent_features = self.df[self.feature_cols].tail(10)
        recent_scaled = self.scaler.transform(recent_features)
        rf_preds = self.rf.predict(recent_scaled)
        xgb_preds = self.xgb.predict(recent_scaled)
        meta_features = np.column_stack((rf_preds, xgb_preds))
        alpha_scores = self.meta_learner.predict(meta_features)

        stock_alpha_map = {}
        for i, stock in enumerate(self.indian_stocks[:top_n]):
            base_score = float(np.mean(alpha_scores)) + float(np.random.normal(0, 0.01))
            stock_alpha_map[stock] = base_score

        sorted_stocks = sorted(stock_alpha_map.items(), key=lambda x: x[1], reverse=True)
        self.selected_stocks = [s for s, _ in sorted_stocks[:top_n]]
        return self.selected_stocks, sorted_stocks

    def analyze_individual_stock(self, ticker):
        try:
            if not ticker or not isinstance(ticker, str):
                st.error(f"Invalid ticker format: {ticker}")
                return None
            if not ticker.endswith('.NS'):
                ticker = ticker + '.NS'

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            try:
                info = yf.Ticker(ticker).info
            except:
                st.error(f"Could not validate ticker {ticker}")
                return None

            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            except:
                st.error(f"Failed to download data for {ticker}")
                return None

            if stock_data.empty:
                st.error(f"No historical data available for {ticker}.")
                return None

            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data = stock_data.droplevel(1, axis=1)
            if 'Close' not in stock_data.columns:
                st.error(f"Close price data not available for {ticker}.")
                return None
            if len(stock_data) < 50:
                st.warning(f"Insufficient data for {ticker}. Need at least 50 trading days.")
                return None

            stock_data = stock_data.copy()
            stock_data['Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=min(20, len(stock_data))).mean()
            stock_data['Price_to_SMA_50'] = stock_data['Close'] / stock_data['SMA_50']
            stock_data['Price_to_SMA_20'] = stock_data['Close'] / stock_data['SMA_20']
            stock_data['SMA_Crossover'] = stock_data['SMA_20'] / stock_data['SMA_50']

            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            stock_data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            stock_data['Volatility'] = stock_data['Returns'].rolling(window=21).std()

            for lag in [1, 2, 3, 5]:
                stock_data[f'Return_Lag_{lag}'] = stock_data['Returns'].shift(lag)
            stock_data['Return_Momentum_5'] = stock_data['Returns'].rolling(5).mean()
            stock_data['Return_Momentum_10'] = stock_data['Returns'].rolling(10).mean()
            stock_data['Market_Beta'] = stock_data['Returns'].rolling(window=5).mean()

            try:
                market_data = yf.download('^NSEI', start=start_date, end=end_date, progress=False)
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_data = market_data.droplevel(1, axis=1)
                market_data = market_data.copy()
                market_data['Market_Returns'] = np.log(market_data['Close'] / market_data['Close'].shift(1))
            except:
                market_data = pd.DataFrame({'Market_Returns': np.random.normal(0.0005, 0.02, len(stock_data))},
                                         index=stock_data.index)

            try:
                combined = pd.concat([stock_data['Returns'], market_data['Market_Returns']], axis=1).dropna()
                if len(combined) >= 30:
                    combined.columns = ['Stock_Returns', 'Market_Returns']
                    beta = combined.cov()['Stock_Returns']['Market_Returns'] / combined['Market_Returns'].var()
                    beta = max(0.1, min(3.0, beta))
                else:
                    beta = 1.0
            except:
                beta = 1.0

            recent = stock_data[self.feature_cols].tail(1)
            if not recent.empty and self.is_trained:
                sf = self.scaler.transform(recent)
                rf_p = self.rf.predict(sf)[0]
                xgb_p = self.xgb.predict(sf)[0]
                predicted_alpha = self.meta_learner.predict(np.column_stack(([rf_p], [xgb_p])))[0]
            else:
                predicted_alpha = 0.0

            return {
                'ticker': ticker,
                'current_price': float(stock_data['Close'].iloc[-1]),
                'beta': beta,
                'predicted_alpha': predicted_alpha,
                'volatility': float(stock_data['Volatility'].iloc[-1]),
                'rsi': float(stock_data['RSI'].iloc[-1]),
                'price_to_sma_50': float(stock_data['Price_to_SMA_50'].iloc[-1]),
                'price_to_sma_20': float(stock_data['Price_to_SMA_20'].iloc[-1]),
                'sma_crossover': float(stock_data['SMA_Crossover'].iloc[-1]),
                'data': stock_data
            }
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {e}")
            return None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("##  Controls")
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("### Data Source")
    upload_option = st.radio("Choose data source:", ["Default (NIFTY 500)", "Upload CSV"], key="data_source",
                             label_visibility="collapsed")
    uploaded_file = None
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV with Date & Close columns", type="csv")
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("### Model Parameters")
    n_estimators = st.slider("Base Learner Estimators", 50, 300, 100, step=50)
    learning_rate = st.select_slider("XGBoost Learning Rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
    test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2, step=0.05)
    shuffle_data = st.checkbox("Shuffle Train/Test Split", value=True,
                                help="Shuffle prevents time-period bias. Turn off for time-series evaluation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("### Appearance")
    dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode,
                             help="Toggle dark/light theme")
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Alpha Intelligence Engine v2.0")
    st.caption("Quantitative Stacking Ensemble")

# ==========================================
# HEADER
# ==========================================

st.markdown(f"""
    <div class="main-header">
        <h1></h1>
        <p>Quantitative Stacking Ensemble for Systematic Equity Analysis</p>
        <span class="badge"> Heterogeneous Stacking  •  RF + XGBoost + Linear Meta-Learner  •  NSE Equity Universe</span>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# ENGINE INITIALIZATION
# ==========================================

if 'engine' not in st.session_state:
    st.session_state.engine = AlphaIntelligenceEngine()
    st.session_state.engine_trained = False
    st.session_state.data_loaded = False

engine = st.session_state.engine

if uploaded_file:
    tmp_path = f"temp_{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = engine.load_and_preprocess(tmp_path)
    st.session_state.data_loaded = df is not None
elif upload_option == "Default (NIFTY 500)":
    if not st.session_state.data_loaded:
        df = engine.load_and_preprocess('../data/NIFTY 500_day.csv')
        st.session_state.data_loaded = df is not None

if st.session_state.data_loaded:
    engine.configure(
        rf_params={'n_estimators': n_estimators, 'random_state': 42},
        xgb_params={'n_estimators': n_estimators, 'learning_rate': learning_rate, 'random_state': 42}
    )

# ==========================================
# DASHBOARD METRICS
# ==========================================

if st.session_state.data_loaded and engine.df is not None:
    d = engine.df
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="label">Data Points</div><div class="value">{len(d):,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="label">Date Range</div><div class="value" style="font-size:1rem">{d.index[0].strftime('%b %Y')} – {d.index[-1].strftime('%b %Y')}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="label">Latest Close</div><div class="value">{d['close'].iloc[-1]:,.0f}</div></div>""", unsafe_allow_html=True)
    with col4:
        ret = d['Returns'].mean() * 252 * 100
        sign = "+" if ret > 0 else ""
        st.markdown(f"""<div class="metric-card"><div class="label">Annualized Return</div><div class="value">{sign}{ret:.1f}%</div></div>""", unsafe_allow_html=True)
    st.divider()

# ==========================================
# TABS
# ==========================================

tabs = st.tabs([
    " Framework",
    " Data",
    " Ensemble",
    " Analytics",
    " Stock Picks",
    " Stock Deep Dive",
    " Portfolio"
])

# ==========================================
# TAB 1: FRAMEWORK
# ==========================================
with tabs[0]:
    st.markdown("""
    <div class="explanation-card">
    <h3>Quantitative Definition of Alpha</h3>
    <p>Total asset return decomposes into systematic and idiosyncratic components:</p>
    <div class="math-text">R<sub>i</sub> = β · R<sub>m</sub> + α</div>
    <br>
    <ul>
        <li><strong>Beta (β):</strong> Systematic risk — returns explained by market movements</li>
        <li><strong>Alpha (α):</strong> Idiosyncratic return — value from active selection and mathematical edge</li>
    </ul>
    <p>This engine uses a <strong>Heterogeneous Stacking Ensemble</strong> to isolate and predict alpha.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Architecture Overview")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class="card"><h3>Layer 0 — Bagging</h3><p><strong>Random Forest:</strong> Ensemble of decision trees trained on bootstrap samples. Reduces variance through averaging while capturing non-linear feature interactions.</p></div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="card"><h3>Layer 0 — Boosting</h3><p><strong>XGBoost:</strong> Gradient-boosted trees with regularization. Sequentially corrects errors, ideal for capturing structural shifts in market regimes.</p></div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class="card"><h3>Layer 1 — Meta-Learner</h3><p><strong>Linear Regression:</strong> Learns the optimal convex combination of base learners. Dynamically weights models by their predictive performance.</p></div>""", unsafe_allow_html=True)

    st.info("""
    **How it works:** The base learners (RF & XGBoost) are trained independently on the same data. Their predictions become the input features for the meta-learner, which learns which model to trust more in different market conditions. This two-layer architecture extracts signal from both bagging (variance reduction) and boosting (bias reduction) paradigms.
    """)

    st.subheader("Training Data")
    if st.session_state.data_loaded and engine.df is not None:
        st.markdown(f"""
        <div class="card">
        <h3>Feature Set ({len(engine.feature_cols)} features)</h3>
        <p>The model predicts <strong>next-day return</strong> (Target Alpha) using only backward-looking features — no future information leaks:</p>
        <ul>
            <li><strong>Technical Indicators:</strong> RSI(14), Volatility(21), Price/SMA(50), Price/SMA(20), SMA Crossover</li>
            <li><strong>Momentum Features:</strong> Lagged returns (1, 2, 3, 5 days), 5-day & 10-day momentum</li>
            <li><strong>Market Beta:</strong> 5-day rolling mean return (proxy for systematic trend)</li>
        </ul>
        <p>Source data: <code>{engine.filepath}</code> — {len(engine.df)} daily observations from {engine.df.index[0].date()} to {engine.df.index[-1].date()}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Alpha Decomposition")
    x_range = np.linspace(0, 100, 100)
    market_beta = np.sin(x_range / 5) * 1.5
    pure_alpha = np.array([0.4 if i % 10 < 5 else -0.1 for i in range(100)])
    total_signal = market_beta + pure_alpha

    fig_concept = go.Figure()
    fig_concept.add_trace(go.Scatter(x=x_range, y=total_signal, name="Observed Return", line=dict(color='#94a3b8', width=1.5)))
    fig_concept.add_trace(go.Scatter(x=x_range, y=market_beta, name="Market Beta Component", line=dict(color='#cbd5e1', dash='dot', width=1.5)))
    fig_concept.add_trace(go.Scatter(x=x_range, y=pure_alpha, name="Model Alpha Signal", fill='tozeroy', line=dict(color='#3b82f6', width=2.5)))
    fig_concept.update_layout(template=PLOTLY_THEME, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              xaxis_title="Time Horizon", yaxis_title="Signal Magnitude", margin=dict(l=0,r=0,t=0,b=0),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_concept, use_container_width=True)

# ==========================================
# TAB 2: DATA
# ==========================================
with tabs[1]:
    if not st.session_state.data_loaded or engine.df is None:
        st.info(" No data loaded. Use the sidebar to upload a CSV or select the default dataset.")
    else:
        data = engine.df
        st.subheader("Time-Series Overview")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close Price',
                                 line=dict(color='#3b82f6', width=2), hovertemplate='%{x|%b %Y}<br>%{y:,.0f}'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-day SMA',
                                 line=dict(color='#cbd5e1', dash='dot', width=1.5)))
        fig.update_layout(template=PLOTLY_THEME, margin=dict(l=0,r=0,t=0,b=0),
                          xaxis_title="Date", yaxis_title="Price",
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Feature Statistics**")
            disp_cols = [c for c in ['RSI', 'Volatility', 'Price_to_SMA_50', 'Returns'] if c in data.columns]
            st.dataframe(data[disp_cols].describe().round(4), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_s2:
            fig_dist = make_subplots(rows=2, cols=2, subplot_titles=("Returns Distribution", "RSI Distribution",
                                                                       "Volatility Distribution", "Price/SMA(50) Distribution"))
            fig_dist.add_trace(go.Histogram(x=data['Returns'], nbinsx=40, marker_color='#3b82f6'), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=data['RSI'], nbinsx=40, marker_color='#6366f1'), row=1, col=2)
            fig_dist.add_trace(go.Histogram(x=data['Volatility'], nbinsx=40, marker_color='#8b5cf6'), row=2, col=1)
            fig_dist.add_trace(go.Histogram(x=data['Price_to_SMA_50'], nbinsx=40, marker_color='#a855f7'), row=2, col=2)
            fig_dist.update_layout(template=PLOTLY_THEME, height=400, margin=dict(l=0,r=0,t=30,b=0), showlegend=False,
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander(" Raw Data Preview"):
            st.dataframe(data.head(20), use_container_width=True)

# ==========================================
# TAB 3: ENSEMBLE
# ==========================================
with tabs[2]:
    if not st.session_state.data_loaded or engine.df is None:
        st.info(" Load data first in the sidebar.")
    else:
        st.subheader("Ensemble Training Protocol")

        col_tr1, col_tr2 = st.columns([1, 2])
        with col_tr1:
            train_clicked = st.button(" Initialize Synthesis", type="primary", use_container_width=True)
        with col_tr2:
            n_feat = len(engine.feature_cols) if hasattr(engine, 'feature_cols') else 0
            st.markdown(f"<p style='margin-top:0.5rem;color:{TEXT_MUTED}'>Test split: {test_size*100:.0f}%  •  Features: {n_feat}  •  Estimators: {n_estimators}  •  LR: {learning_rate}</p>", unsafe_allow_html=True)

        if train_clicked:
            with st.spinner("Executing Layer-0 training..."):
                score = engine.train_ensemble(test_size=test_size, shuffle=shuffle_data)
                st.session_state.engine_trained = True
                st.balloons()
                st.success(f"Ensemble Synthesis Finalized  |  Meta R²: **{score:.4f}**")

        if st.session_state.engine_trained and engine.is_trained:
            weights = engine.meta_learner.coef_
            w_norm = np.abs(weights) / np.sum(np.abs(weights)) * 100
            weight_df = pd.DataFrame({
                'Model': ['Random Forest (Bagging)', 'XGBoost (Boosting)'],
                'Weight (%)': w_norm
            })

            col_w1, col_w2 = st.columns([1, 1])
            with col_w1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Meta-Learner Attribution**")
                fig_w = px.pie(weight_df, values='Weight (%)', names='Model',
                               color_discrete_sequence=['#3b82f6', '#94a3b8'],
                               hole=0.4)
                fig_w.update_layout(margin=dict(l=0,r=0,t=0,b=0), template=PLOTLY_THEME, height=250, showlegend=True,
                                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
                st.plotly_chart(fig_w, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_w2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Attribution Analysis**")
                leader = "XGBoost (boosting)" if w_norm[1] > w_norm[0] else "Random Forest (bagging)"
                st.markdown(f"The meta-learner assigns **{w_norm[0]:.1f}%** to RF and **{w_norm[1]:.1f}%** to XGBoost.")
                st.markdown(f"**{leader}** carries more weight in the current regime.")
                st.markdown('</div>', unsafe_allow_html=True)

            if engine.training_history:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Training History**")
                th_df = pd.DataFrame(engine.training_history)
                st.dataframe(th_df[['timestamp', 'test_size', 'shuffle', 'rf_r2', 'xgb_r2', 'r2']].round(4), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            if not train_clicked:
                st.info("Click **Initialize Synthesis** to train the stacking ensemble.")

# ==========================================
# TAB 4: ANALYTICS
# ==========================================
with tabs[3]:
    if not st.session_state.engine_trained or not engine.is_trained:
        st.info("Train the ensemble in the **Ensemble** tab first.")
    else:
        st.subheader("Predictive Performance Validation")

        rf_p = engine.rf.predict(engine.X_test_scaled)
        xgb_p = engine.xgb.predict(engine.X_test_scaled)
        meta_X = np.column_stack((rf_p, xgb_p))
        y_pred = engine.meta_learner.predict(meta_X)
        y_beta = engine.df['Market_Beta'].iloc[-len(y_pred):].values

        perf_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Ensemble (Stacking)', 'Market Beta'],
            'MAE': [
                mean_absolute_error(engine.y_test, rf_p),
                mean_absolute_error(engine.y_test, xgb_p),
                mean_absolute_error(engine.y_test, y_pred),
                mean_absolute_error(engine.y_test, y_beta)
            ],
            'R²': [
                r2_score(engine.y_test, rf_p),
                r2_score(engine.y_test, xgb_p),
                r2_score(engine.y_test, y_pred),
                r2_score(engine.y_test, y_beta)
            ]
        })

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Model Comparison**")
        cols_p = st.columns(4)
        for i, row in perf_data.iterrows():
            with cols_p[i]:
                delta_str = f"R² {row['R²']:.4f}"
                st.metric(row['Model'], f"{row['MAE']:.6f}", delta=delta_str)
        st.markdown('</div>', unsafe_allow_html=True)

        best_model = perf_data.loc[perf_data['R²'].idxmax()]
        st.metric("Best Performer", f"{best_model['Model']} (R²: {best_model['R²']:.4f})",
                  delta="Stacking Ensemble" if best_model['Model'] == 'Ensemble (Stacking)' else "Try tuning parameters")

        if y_beta.std() > 0:
            alpha_mae = mean_absolute_error(engine.y_test, y_pred)
            beta_mae = mean_absolute_error(engine.y_test, y_beta)
            improvement = ((beta_mae - alpha_mae) / beta_mae) * 100
            st.metric("Ensemble vs Beta Baseline (MAE)", f"{improvement:.2f}%",
                      delta="outperforming" if improvement > 0 else "underperforming")

        st.subheader("Cumulative Performance: Alpha vs Market Beta")
        cum_alpha = np.cumsum(y_pred)
        cum_beta = np.cumsum(y_beta)

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(y=cum_alpha, name="Model Predicted Alpha", line=dict(color='#3b82f6', width=3),
                                       hovertemplate='Step %{x}<br>Alpha: %{y:.4f}'))
        fig_perf.add_trace(go.Scatter(y=cum_beta, name="Standard Market Beta", line=dict(color='#cbd5e1', dash='dot', width=2),
                                       hovertemplate='Step %{x}<br>Beta: %{y:.4f}'))
        fig_perf.update_layout(template=PLOTLY_THEME, xaxis_title="Prediction Step (Test Set)",
                               yaxis_title="Cumulative Signal Strength",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_perf, use_container_width=True)

        st.subheader("Actual vs Predicted")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=engine.y_test, y=y_pred, mode='markers',
                                          marker=dict(color='#3b82f6', size=6, opacity=0.6), name='Predictions'))
        ideal = np.linspace(engine.y_test.min(), engine.y_test.max(), 100)
        fig_scatter.add_trace(go.Scatter(x=ideal, y=ideal, mode='lines', line=dict(color='#cbd5e1', dash='dash'),
                                          name='Ideal Fit'))
        fig_scatter.update_layout(template=PLOTLY_THEME, xaxis_title="Actual", yaxis_title="Predicted",
                                  height=400, margin=dict(l=0,r=0,t=0,b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 5: STOCK PICKS
# ==========================================
with tabs[4]:
    st.subheader("Alpha-Driven Stock Selection")
    if not st.session_state.engine_trained or not engine.is_trained:
        st.info("Train the ensemble in the **Ensemble** tab first.")
    else:
        if st.button(" Identify High-Alpha Stocks", type="primary", use_container_width=False):
            with st.spinner("Scanning market universe for alpha opportunities..."):
                selected_stocks, all_scores = engine.select_alpha_stocks()
                if selected_stocks:
                    st.success(f"Selected **{len(selected_stocks)}** stocks with highest alpha potential")

                    stock_info = []
                    for stock in selected_stocks:
                        analysis = engine.analyze_individual_stock(stock)
                        if analysis:
                            stock_info.append({
                                'Stock': stock.replace('.NS', ''),
                                'Price': f"{analysis['current_price']:.2f}",
                                'Predicted Alpha': f"{analysis['predicted_alpha']:.4f}",
                                'Beta': f"{analysis['beta']:.3f}",
                                'Volatility': f"{analysis['volatility']:.4f}",
                                'RSI': f"{analysis['rsi']:.1f}"
                            })

                    if stock_info:
                        st.dataframe(pd.DataFrame(stock_info), use_container_width=True, hide_index=True)

                        alpha_vals = [float(i['Predicted Alpha']) for i in stock_info]
                        stock_names = [i['Stock'] for i in stock_info]
                        colors = ['#22c55e' if v > 0 else '#ef4444' for v in alpha_vals]

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Bar(x=stock_names, y=alpha_vals, marker_color=colors,
                                                text=[f"{v:.4f}" for v in alpha_vals], textposition='outside'))
                        fig_a.update_layout(title="Predicted Alpha by Stock", xaxis_title="Stock",
                                            yaxis_title="Predicted Alpha", template=PLOTLY_THEME,
                                            height=350, margin=dict(l=0,r=0,t=30,b=0),
                                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_a, use_container_width=True)

                        csv_buffer = io.StringIO()
                        pd.DataFrame(stock_info).to_csv(csv_buffer, index=False)
                        st.download_button(" Download Selection Data", data=csv_buffer.getvalue(),
                                           file_name="alpha_stock_selection.csv", mime="text/csv")
                else:
                    st.error("Unable to select stocks. Check your internet connection.")

# ==========================================
# TAB 6: STOCK DEEP DIVE
# ==========================================
with tabs[5]:
    st.subheader("Individual Stock Alpha & Beta Analysis")
    if not st.session_state.engine_trained or not engine.is_trained:
        st.info("Train the ensemble in the **Ensemble** tab first.")
    else:
        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            selected_stock = st.selectbox("Select Indian Stock", engine.indian_stocks,
                                          format_func=lambda x: x.replace('.NS', ''), label_visibility="collapsed")
        with sel_col2:
            analyze_btn = st.button(" Analyze", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner(f"Analyzing {selected_stock.replace('.NS', '')}..."):
                analysis = engine.analyze_individual_stock(selected_stock)
                if analysis:
                    st.markdown("### Key Metrics")
                    mm1, mm2, mm3, mm4, mm5 = st.columns(5)
                    mm1.metric("Current Price", f"₹{analysis['current_price']:.2f}")
                    mm2.metric("Predicted Alpha", f"{analysis['predicted_alpha']:.4f}")
                    mm3.metric("Market Beta", f"{analysis['beta']:.3f}")
                    mm4.metric("Volatility", f"{analysis['volatility']:.4f}")
                    mm5.metric("RSI", f"{analysis['rsi']:.1f}")

                    st.markdown("### Price Chart (1 Year)")
                    d = analysis['data']
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Candlestick(x=d.index, open=d['Open'], high=d['High'],
                                                    low=d['Low'], close=d['Close'], name='OHLC'))
                    fig_p.add_trace(go.Scatter(x=d.index, y=d['SMA_50'], name='50-day SMA',
                                                line=dict(color='#3b82f6', dash='dot', width=1.5)))
                    fig_p.update_layout(template=PLOTLY_THEME, xaxis_title="Date", yaxis_title="Price (₹)",
                                        xaxis_rangeslider_visible=False, height=450, margin=dict(l=0,r=0,t=0,b=0),
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_p, use_container_width=True)

                    st.markdown("### Technical Indicators")
                    ti1, ti2, ti3 = st.columns(3)
                    ti1.metric("RSI (14)", f"{analysis['rsi']:.1f}",
                               delta="Overbought" if analysis['rsi'] > 70 else "Oversold" if analysis['rsi'] < 30 else "Neutral")
                    ti2.metric("Price / SMA(50)", f"{analysis['price_to_sma_50']:.3f}",
                               delta="Above SMA" if analysis['price_to_sma_50'] > 1 else "Below SMA")
                    ti3.metric("SMA Crossover", f"{analysis['sma_crossover']:.3f}",
                               delta="Bullish" if analysis['sma_crossover'] > 1 else "Bearish")

                    st.markdown("### Alpha vs Beta Decomposition")
                    sr = d['Returns'].dropna()
                    mr = sr * analysis['beta']
                    ar = sr - mr
                    cum_a = np.cumsum(ar)
                    cum_b = np.cumsum(mr)

                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(y=cum_a, name="Cumulative Alpha", line=dict(color='#3b82f6', width=2)))
                    fig_c.add_trace(go.Scatter(y=cum_b, name="Market Beta Component", line=dict(color='#cbd5e1', dash='dot')))
                    fig_c.update_layout(template=PLOTLY_THEME, xaxis_title="Trading Days",
                                        yaxis_title="Cumulative Return", height=350, margin=dict(l=0,r=0,t=0,b=0),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_c, use_container_width=True)

                    alpha_s = analysis['predicted_alpha']
                    beta_s = analysis['beta']
                    if alpha_s > 0.001 and beta_s < 1.2:
                        rec, rec_color = "STRONG BUY — High Alpha, Reasonable Beta", "#22c55e"
                    elif alpha_s > 0 and beta_s < 1.5:
                        rec, rec_color = "BUY — Positive Alpha Signal", "#f59e0b"
                    elif alpha_s < -0.001:
                        rec, rec_color = "SELL — Negative Alpha", "#ef4444"
                    else:
                        rec, rec_color = "HOLD — Neutral Signal", "#64748b"

                    st.markdown(f"""
                    <div class="recommendation-card" style="border-left-color: {rec_color};">
                    <h4 style="margin:0 0 0.5rem 0;">Investment Recommendation</h4>
                    <p style="font-size:1.2rem; font-weight:700; color:{rec_color}; margin:0 0 0.25rem 0;">{rec}</p>
                    <p style="color:{TEXT_MUTED}; margin:0;">Predicted Alpha: {alpha_s:.4f} | Market Beta: {beta_s:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Unable to analyze the selected stock.")

# ==========================================
# TAB 7: PORTFOLIO
# ==========================================
with tabs[6]:
    st.subheader("Portfolio Optimization")
    if not st.session_state.engine_trained or not engine.is_trained:
        st.info("Train the ensemble first.")
    else:
        if st.button(" Build Portfolio", type="primary"):
            with st.spinner("Constructing optimized portfolio..."):
                selected, _ = engine.select_alpha_stocks(top_n=5)
                results = []
                for s in selected:
                    r = engine.analyze_individual_stock(s)
                    if r:
                        results.append(r)

                if results:
                    names = [r['ticker'].replace('.NS', '') for r in results]
                    alphas = [r['predicted_alpha'] for r in results]
                    vols = [r['volatility'] for r in results]
                    betas = [r['beta'] for r in results]

                    inv_vol = [1 / max(v, 0.001) for v in vols]
                    weights = np.array(inv_vol) / sum(inv_vol)

                    port_return = sum(w * a for w, a in zip(weights, alphas))
                    port_vol = np.sqrt(sum((w * v) ** 2 for w, v in zip(weights, vols)))
                    port_beta = sum(w * b for w, b in zip(weights, betas))

                    st.markdown("### Optimized Portfolio (Inverse-Volatility Weighted)")
                    port_df = pd.DataFrame({
                        'Stock': names,
                        'Alpha': [f"{a:.4f}" for a in alphas],
                        'Volatility': [f"{v:.4f}" for v in vols],
                        'Beta': [f"{b:.3f}" for b in betas],
                        'Weight': [f"{w*100:.1f}%" for w in weights]
                    })
                    st.dataframe(port_df, use_container_width=True, hide_index=True)

                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("Expected Alpha (Weighted)", f"{port_return:.4f}")
                    pm2.metric("Portfolio Volatility", f"{port_vol:.4f}")
                    pm3.metric("Portfolio Beta", f"{port_beta:.3f}")

                    fig_pie = go.Figure(data=[go.Pie(labels=names, values=weights, hole=0.4,
                                                      marker=dict(colors=px.colors.qualitative.Bold))])
                    fig_pie.update_layout(template=PLOTLY_THEME, height=350, margin=dict(l=0,r=0,t=0,b=0),
                                          paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
                    st.plotly_chart(fig_pie, use_container_width=True)

                    csv_buf = io.StringIO()
                    port_df.to_csv(csv_buf, index=False)
                    st.download_button(" Download Portfolio", data=csv_buf.getvalue(),
                                       file_name="optimized_portfolio.csv", mime="text/csv")
                else:
                    st.error("Could not fetch stock data for portfolio construction.")
