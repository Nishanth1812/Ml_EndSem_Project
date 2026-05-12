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
├── api
│   ├── main.py
│   ├── run.py
│   ├── schemas.py
│   └── service.py
├── frontend
│   ├── app
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   ├── metrics
│   │   │   └── page.tsx
│   │   ├── inference
│   │   │   └── page.tsx
│   │   └── screener
│   │       └── page.tsx
│   ├── components
│   │   └── sidebar-nav.tsx
│   └── lib
│       ├── api.ts
│       └── types.ts
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
- The Next.js UI is now dark, simple, and page-wise.
- The frontend is limited to metrics display, inference, and watchlist ranking.

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

## New Web Stack

- `api/` contains the FastAPI backend.
- `frontend/` contains the Next.js dashboard.
- Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the backend is not on `http://localhost:8000`.
- Start the backend with `python -m api.run`.
- Start the frontend with `npm run dev` inside `frontend/`.
- Pages available in the frontend: `Overview`, `Metrics`, `Inference`, and `Screener`.

## Model flow
1. Load the CSV price history.
2. Standardize column names and engineer lag, momentum, RSI, volatility, and price-distance features.
3. Train the baseline linear model on scaled features.
4. Tune and train Random Forest and XGBoost separately.
5. Use one model at a time for stock scoring and ranking.

## Dependencies
Install the packages listed in `requirements.txt` or via the `pyproject.toml` metadata.