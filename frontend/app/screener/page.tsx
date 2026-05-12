"use client";

import { useState } from 'react';

import { fetchJson } from '../../lib/api';
import { MODEL_OPTIONS, type RankResponse } from '../../lib/types';
import { getRecommendation } from '../../lib/insights';

const DEFAULT_TICKERS = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY'];

export default function Page() {
  const [modelName, setModelName] = useState<(typeof MODEL_OPTIONS)[number]['value']>('xgboost');
  const [items, setItems] = useState<RankResponse['items']>([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleRank() {
    setLoading(true);
    setMessage('');
    try {
      const payload = await fetchJson<RankResponse>('/rank', {
        method: 'POST',
        body: JSON.stringify({ model_name: modelName, top_n: 5, tickers: DEFAULT_TICKERS, lookback_days: 365 }),
      });
      setItems(payload.items.sort((left, right) => right.predicted_alpha - left.predicted_alpha));
      setMessage(`Ranked ${payload.items.length} stocks.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Ranking failed');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  const topItem = items[0] ?? null;
  const averageAlpha = items.length > 0 ? items.reduce((sum, item) => sum + item.predicted_alpha, 0) / items.length : null;
  const averageBeta = items.length > 0 ? items.reduce((sum, item) => sum + item.beta, 0) / items.length : null;
  const lowestBeta = items.length > 0 ? items.reduce((best, item) => (item.beta < best.beta ? item : best), items[0]) : null;

  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Stock Screener</h1>
          <p className="subtitle">
            Rank your watchlist by predicted performance. Select a model and run the screener to see the top stocks.
          </p>
        </div>

        <div className="hero-aside hero-summary-grid">
          <div className="status">
            <strong>Watchlist</strong>
            <span>{DEFAULT_TICKERS.length} stocks</span>
          </div>
          <div className="status">
            <strong>Results</strong>
            <span>{items.length > 0 ? items.length : '—'}</span>
          </div>
        </div>
      </section>

      <section className="grid two">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Ranking controls</div>
              <h2>Watchlist scoring</h2>
            </div>
            <div className="panel-note">Read-only</div>
          </div>
        <p className="subtitle">
          Rank a small universe by predicted alpha using the selected inference model. Training controls are intentionally absent from the UI.
        </p>

          <div className="controls">
            <div className="field">
              <label htmlFor="ranking-model">Model</label>
              <select id="ranking-model" value={modelName} onChange={(event) => setModelName(event.target.value as (typeof MODEL_OPTIONS)[number]['value'])}>
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="actions">
              <button className="btn btn-primary" onClick={handleRank} type="button" disabled={loading}>
                {loading ? 'Ranking...' : 'Rank watchlist'}
              </button>
              <button className="btn btn-secondary" onClick={() => setItems([])} type="button" disabled={loading}>
                Clear
              </button>
            </div>
          </div>

          {message ? <div className="status" style={{ marginTop: 14 }}>{message}</div> : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Results</div>
              <h2>Alpha ranking</h2>
            </div>
            {topItem ? <div className={`signal-chip tone-${getRecommendation(topItem.predicted_alpha, topItem.beta).tone}`}>Top pick</div> : null}
          </div>

          <div className="summary-strip" style={{ marginTop: 12 }}>
            <div className="summary-card">
              <span>Top ticker</span>
              <strong>{topItem ? topItem.ticker.replace('.NS', '') : '—'}</strong>
              <small>{topItem ? `Alpha ${topItem.predicted_alpha.toFixed(5)}` : 'Run ranking to populate the table.'}</small>
            </div>
            <div className="summary-card">
              <span>Average alpha</span>
              <strong>{averageAlpha !== null ? averageAlpha.toFixed(5) : '—'}</strong>
              <small>Average edge across the ranked watchlist.</small>
            </div>
            <div className="summary-card">
              <span>Lowest beta</span>
              <strong>{lowestBeta ? lowestBeta.ticker.replace('.NS', '') : '—'}</strong>
              <small>{averageBeta !== null ? `Average beta ${averageBeta.toFixed(3)}` : 'Run ranking to see risk context.'}</small>
            </div>
          </div>

          {items.length > 0 ? (
            <div className="table-shell" style={{ marginTop: 16 }}>
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Signal</th>
                    <th>Alpha</th>
                    <th>Return</th>
                    <th>Beta</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, index) => {
                    const recommendation = getRecommendation(item.predicted_alpha, item.beta);

                    return (
                      <tr key={item.ticker}>
                        <td><span className="rank-pill">{index + 1}</span></td>
                        <td>
                          <div className="table-primary">{item.ticker.replace('.NS', '')}</div>
                          <small>{item.model_name}</small>
                        </td>
                        <td><span className={`signal-chip tone-${recommendation.tone}`}>{recommendation.label}</span></td>
                        <td>
                          <div className="table-primary">{item.predicted_alpha.toFixed(5)}</div>
                          <div className="table-meter"><span style={{ width: `${Math.min(100, Math.max(18, (item.predicted_alpha + 0.01) * 900))}%` }} /></div>
                        </td>
                        <td>{item.predicted_return.toFixed(5)}</td>
                        <td>{item.beta.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="section-label">No ranking yet</div>
              <h3>Rank the watchlist to populate the table.</h3>
              <p>Once you run the request, the UI will show alpha ordering, signal labels, and the model-specific risk context.</p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}