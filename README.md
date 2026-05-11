# Alpha Intelligence Engine

## Overview
The project now uses a clearer supervised-learning layout:

- Linear Regression is the baseline model.
- Random Forest and XGBoost are the primary inference models.
- Each model is trained independently and used directly for prediction.
- Alpha is derived as the model return forecast adjusted by estimated beta and market expectation.

## Structure
```
Ml_EndSem_Project
├── main.py
├── pyproject.toml
├── requirements.txt
├── src
│   ├── myapp.py
│   ├── run_app.py
│   └── alpha_engine
│       ├── __init__.py
│       ├── engine.py
│       └── ui.py
├── data
│   └── NIFTY 500_day.csv
└── setup.sh
```

## What changed
- The old stacked ensemble was removed.
- Preprocessing, training, inference, and UI code are now separated.
- Chronological train/test splitting is used to avoid leakage.
- Random Forest and XGBoost are tuned independently and can be selected at inference time.

## Run the app
Use any of the following:

```bash
streamlit run src/myapp.py
```

```bash
python main.py
```

```bash
python src/run_app.py
```

## Model flow
1. Load the CSV price history.
2. Standardize column names and engineer lag, momentum, RSI, volatility, and price-distance features.
3. Train the baseline linear model on scaled features.
4. Tune and train Random Forest and XGBoost separately.
5. Use one model at a time for stock scoring and ranking.

## Dependencies
Install the packages listed in `requirements.txt` or via the `pyproject.toml` metadata.