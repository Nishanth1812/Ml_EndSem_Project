import Link from 'next/link';

import { NAV_ITEMS } from '../lib/types';

const PRODUCT_NOTES: Record<string, string> = {
  '/data': 'Load the market dataset and review the cleaned preview used by the engine.',
  '/training': 'Run chronological training and review holdout metrics and importance.',
  '/inference': 'Score a ticker with Linear Regression, Random Forest, or XGBoost.',
  '/screener': 'Rank the watchlist by predicted alpha in a compact table view.',
};

export default function Page() {
  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Stock Analysis</h1>
          <p className="subtitle">
            Analyze stocks using machine learning models. View historical data, training metrics, individual predictions, and watchlist rankings.
          </p>

          <div className="actions" style={{ marginTop: 18 }}>
            {NAV_ITEMS.map((item) => (
              <Link key={item.href} href={item.href} className="btn btn-secondary">
                {item.label}
              </Link>
            ))}
          </div>
        </div>

        <div className="hero-aside panel panel-compact">
          <div className="section-label">Models</div>
          <div className="stack-list">
            <div className="stack-item">
              <strong>Linear Regression</strong>
              <small>Baseline model</small>
            </div>
            <div className="stack-item">
              <strong>Random Forest</strong>
              <small>Primary model</small>
            </div>
            <div className="stack-item">
              <strong>XGBoost</strong>
              <small>Primary model</small>
            </div>
          </div>
        </div>
      </section>

      <section className="grid two">
        <div className="panel">
          <div className="section-label">Navigation</div>
          <h2>Pages</h2>
          <div className="page-grid">
            {NAV_ITEMS.map((item) => (
              <Link key={item.href} href={item.href} className="page-card">
                <div>
                  <div className="page-card-title">{item.label}</div>
                  <p>{PRODUCT_NOTES[item.href]}</p>
                </div>
                <span className="page-card-link">Open</span>
              </Link>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-label">About</div>
          <h2>Overview</h2>
          <div className="stack-list">
            <div className="stack-item">
              <strong>Data</strong>
              <small>Review historical stock data used by models</small>
            </div>
            <div className="stack-item">
              <strong>Training</strong>
              <small>View model training metrics and performance</small>
            </div>
            <div className="stack-item">
              <strong>Inference</strong>
              <small>Score individual tickers with different models</small>
            </div>
            <div className="stack-item">
              <strong>Screener</strong>
              <small>Rank your watchlist by predicted performance</small>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
