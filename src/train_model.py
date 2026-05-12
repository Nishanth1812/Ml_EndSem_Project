"""
Alpha Intelligence Engine — Model Training
===========================================
Fetches Nifty 100 data, engineers features, trains LR + XGBoost,
runs walk-forward backtest with monthly rebalancing,
saves model for Streamlit inference.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf, pickle, os, time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

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

NIFTY_INDEX = '^NSEI'
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'RSI', 'Volatility',
    'Price_to_SMA_50', 'Price_to_SMA_20', 'SMA_Crossover',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5',
    'Return_Momentum_5', 'Return_Momentum_10', 'Return_Momentum_21',
    'Rolling_Beta', 'Market_Return_Lag_1',
    'Excess_Return_5', 'Excess_Return_10', 'Excess_Return_21',
    'Price_vs_52w_High',
    'Day_of_Week', 'Month_of_Year',
]

print("=" * 60)
print("ALPHA INTELLIGENCE ENGINE -- MODEL TRAINING")
print("=" * 60)

# =========== FETCH DATA ===========
print("\n[1/5] Fetching data from Yahoo Finance...")
end = datetime.now()
start = end - timedelta(days=730)

nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
if isinstance(nifty.columns, pd.MultiIndex):
    nifty = nifty.droplevel(1, axis=1)
nifty['Market_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))

all_dfs = []
failed = []

for i, ticker in enumerate(NIFTY_100):
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(NIFTY_100)}]")
    try:
        d = yf.download(ticker, start=start, end=end, progress=False)
        if d.empty:
            failed.append(ticker); continue
        if isinstance(d.columns, pd.MultiIndex):
            d = d.droplevel(1, axis=1)
        if 'Close' not in d.columns or len(d) < 200:
            failed.append(ticker); continue

        df = d.copy()
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Price_to_SMA_50'] = df['Close'] / df['SMA_50']
        df['Price_to_SMA_20'] = df['Close'] / df['SMA_20']
        df['SMA_Crossover'] = df['SMA_20'] / df['SMA_50']

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['Volatility'] = df['Returns'].rolling(21).std()

        for lag in [1, 2, 3, 5]:
            df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
        df['Return_Momentum_5'] = df['Returns'].rolling(5).mean()
        df['Return_Momentum_10'] = df['Returns'].rolling(10).mean()
        df['Return_Momentum_21'] = df['Returns'].rolling(21).mean()

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

        # Alpha target: ~1 month forward (21 trading days)
        df['Target_Alpha'] = df['Alpha'].shift(-21).rolling(21).sum()

        df['Ticker'] = ticker.replace('.NS', '')
        all_dfs.append(df.dropna(subset=FEATURE_COLS + ['Target_Alpha']))
    except Exception as e:
        failed.append(ticker)

pooled = pd.concat(all_dfs, ignore_index=True)
n_stocks = len(all_dfs)
n_obs = len(pooled)
print(f"  Fetched {n_stocks}/{len(NIFTY_100)} stocks, {n_obs:,} observations")
if failed:
    print(f"  Failed: {failed}")

# =========== WALK-FORWARD BACKTEST ===========
print("\n[2/5] Running walk-forward backtest (monthly rebalance)...")

data = pooled[FEATURE_COLS + ['Target_Alpha', 'Ticker']].dropna().copy()
X_all = data[FEATURE_COLS].values
y_all = data['Target_Alpha'].values

N = len(X_all)
# Train: 60%, Test1: 20%, Test2: 20%
w1 = int(N * 0.6)
w2 = int(N * 0.8)

# --- Period 1 ---
s1 = StandardScaler()
X_tr1 = s1.fit_transform(X_all[:w1])
X_te1 = s1.transform(X_all[w1:w2])
y_tr1 = y_all[:w1]
y_te1 = y_all[w1:w2]

lr1 = LinearRegression().fit(X_tr1, y_tr1)
xgb1 = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbosity=0).fit(X_tr1, y_tr1)
lp1 = lr1.predict(X_te1)
xp1 = xgb1.predict(X_te1)

# --- Period 2 ---
s2 = StandardScaler()
X_tr2 = s2.fit_transform(X_all[:w2])
X_te2 = s2.transform(X_all[w2:])
y_tr2 = y_all[:w2]
y_te2 = y_all[w2:]

lr2 = LinearRegression().fit(X_tr2, y_tr2)
xgb2 = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbosity=0).fit(X_tr2, y_tr2)
lp2 = lr2.predict(X_te2)
xp2 = xgb2.predict(X_te2)

lr_pred = np.concatenate([lp1, lp2])
xgb_pred = np.concatenate([xp1, xp2])
y_act = np.concatenate([y_te1, y_te2])

lr_r2 = r2_score(y_act, lr_pred)
xgb_r2 = r2_score(y_act, xgb_pred)
lr_mae = mean_absolute_error(y_act, lr_pred)
xgb_mae = mean_absolute_error(y_act, xgb_pred)
lr_rho = spearmanr(y_act, lr_pred)[0]
xgb_rho = spearmanr(y_act, xgb_pred)[0]

print(f"  Walk-Forward Performance:")
print(f"  {'':30s} {'R2':>8s} {'MAE':>12s} {'Spearman':>10s}")
print(f"  {'-'*60}")
print(f"  {'Linear Regression (Baseline)':30s} {lr_r2:>8.4f} {lr_mae:>12.6f} {lr_rho:>10.4f}")
print(f"  {'XGBoost':30s} {xgb_r2:>8.4f} {xgb_mae:>12.6f} {xgb_rho:>10.4f}")

best_name = 'XGBoost' if xgb_r2 >= lr_r2 else 'LinearRegression'

# =========== STRATEGY BACKTEST ===========
print("\n[3/5] Backtesting long-short strategy...")

n_batches = 30
batch_size = len(y_act) // n_batches

def backtest_model(preds, actuals, n_batches=30):
    batch_sz = len(actuals) // n_batches
    results = []
    for b in range(n_batches):
        sidx = b * batch_sz
        eidx = sidx + batch_sz if b < n_batches - 1 else len(actuals)
        # Within each batch, sort by predicted alpha
        order = np.argsort(-preds[sidx:eidx])
        batch_preds = preds[sidx:eidx][order]
        batch_actuals = actuals[sidx:eidx][order]
        n_top = max(1, len(batch_actuals) // 5)
        n_bot = max(1, len(batch_actuals) // 5)
        long_ret = np.mean(batch_actuals[:n_top])
        short_ret = np.mean(batch_actuals[-n_bot:])
        spread = long_ret - short_ret
        results.append(spread)
    return np.array(results)

lr_spreads = backtest_model(lr_pred, y_act, n_batches)
xgb_spreads = backtest_model(xgb_pred, y_act, n_batches)
best_spreads = xgb_spreads if xgb_r2 >= lr_r2 else lr_spreads

def compute_strat_metrics(spreads, periods_per_year=12):
    avg = np.mean(spreads)
    std = np.std(spreads) + 1e-9
    sharpe = avg / std * np.sqrt(periods_per_year)
    win_rate = np.mean(spreads > 0)
    cum = np.cumsum(spreads)
    dd = np.maximum.accumulate(cum) - cum
    max_dd = np.max(dd)
    return {
        'avg_spread': float(avg),
        'std_spread': float(std),
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_dd),
        'cumulative': cum.tolist(),
        'spreads': spreads.tolist(),
    }

bt_lr = compute_strat_metrics(lr_spreads)
bt_xgb = compute_strat_metrics(xgb_spreads)
bt_best = compute_strat_metrics(best_spreads)

print(f"  Long-Short Strategy (Monthly Rebalance):")
print(f"  {'':30s} {'LR':>12s} {'XGB':>12s} {'Best':>12s}")
print(f"  {'-'*66}")
print(f"  {'Avg Monthly Alpha Spread':30s} {bt_lr['avg_spread']:>12.6f} {bt_xgb['avg_spread']:>12.6f} {bt_best['avg_spread']:>12.6f}")
print(f"  {'Sharpe Ratio (ann.)':30s} {bt_lr['sharpe']:>12.2f} {bt_xgb['sharpe']:>12.2f} {bt_best['sharpe']:>12.2f}")
print(f"  {'Win Rate':30s} {bt_lr['win_rate']:>12.1%} {bt_xgb['win_rate']:>12.1%} {bt_best['win_rate']:>12.1%}")
print(f"  {'Max Drawdown':30s} {bt_lr['max_drawdown']:>12.6f} {bt_xgb['max_drawdown']:>12.6f} {bt_best['max_drawdown']:>12.6f}")

# =========== TRAIN FINAL MODELS ===========
print("\n[4/5] Training final models on all data...")

scaler_final = StandardScaler()
X_all_s = scaler_final.fit_transform(X_all)

lr_final = LinearRegression().fit(X_all_s, y_all)
xgb_final = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbosity=0).fit(X_all_s, y_all)

# =========== SAVE ===========
print("[5/5] Saving model...")

model = {
    'scaler': scaler_final,
    'linear_regression': lr_final,
    'xgboost': xgb_final,
    'feature_cols': FEATURE_COLS,
    'lr_r2': float(lr_r2),
    'xgb_r2': float(xgb_r2),
    'lr_spearman': float(lr_rho),
    'xgb_spearman': float(xgb_rho),
    'lr_mae': float(lr_mae),
    'xgb_mae': float(xgb_mae),
    'best_model_name': best_name,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_stocks': n_stocks,
    'n_training_samples': len(X_all),
    'target': 'Target_Alpha (21-day forward sum)',
    'feature_cols': FEATURE_COLS,
    'backtest': {
        'lr': bt_lr,
        'xgb': bt_xgb,
        'best': bt_best,
        'selected_model': best_name,
        'periods_per_year': 12,
    }
}

with open(f'{MODELS_DIR}/alpha_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"  Saved to {MODELS_DIR}/alpha_model.pkl")
print(f"  Best model: {best_name}")
print(f"  Walk-Forward R2: {max(lr_r2, xgb_r2):.4f}")
print(f"  Strategy Sharpe: {bt_best['sharpe']:.2f}")
print(f"  Win Rate: {bt_best['win_rate']:.1%}")
print()
print("Done! Run the Streamlit app to see results.")
