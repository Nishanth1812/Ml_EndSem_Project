"use client";

import { useState } from 'react';

import { fetchJson } from '../../lib/api';
import { MODEL_OPTIONS, type AnalysisResponse } from '../../lib/types';
import { getRecommendation } from '../../lib/insights';

export default function Page() {
  const [ticker, setTicker] = useState('RELIANCE');
  const [modelName, setModelName] = useState<(typeof MODEL_OPTIONS)[number]['value']>('xgboost');
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleAnalyze() {
    setLoading(true);
    setMessage('');
    try {
      const payload = await fetchJson<AnalysisResponse>('/analyze', {
        method: 'POST',
        body: JSON.stringify({ ticker, model_name: modelName, lookback_days: 365 }),
      });
      setAnalysis(payload);
      setMessage(`Scored ${payload.ticker} with ${payload.model_name}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Analysis failed');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }

  const recommendation = analysis ? getRecommendation(analysis.predicted_alpha, analysis.beta) : null;

  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Stock Score</h1>
          <p className="subtitle">
            Enter a ticker and select a model to get a prediction. Results show the model's predicted return, risk metrics, and a recommendation.
          </p>
        </div>
      </section>

      <section className="grid two">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Input</div>
              <h2>Ticker</h2>
            </div>
          </div>
        <p className="subtitle">
          Select a ticker and model, then run inference to see results.
        </p>

          <div className="controls">
            <div className="field">
              <label htmlFor="ticker">Ticker</label>
              <input id="ticker" value={ticker} onChange={(event) => setTicker(event.target.value)} placeholder="RELIANCE" />
            </div>
            <div className="field">
              <label htmlFor="model">Model</label>
              <select id="model" value={modelName} onChange={(event) => setModelName(event.target.value as (typeof MODEL_OPTIONS)[number]['value'])}>
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="actions">
              <button className="btn btn-primary" onClick={handleAnalyze} type="button" disabled={loading}>
                {loading ? 'Analyzing...' : 'Run inference'}
              </button>
              <button className="btn btn-secondary" onClick={() => setTicker('RELIANCE')} type="button" disabled={loading}>
                Reset
              </button>
            </div>
          </div>

          {message ? <div className="status" style={{ marginTop: 14 }}>{message}</div> : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Result</div>
              <h2>Score</h2>
            </div>
            {analysis ? <div className={`signal-chip tone-${recommendation?.tone ?? 'neutral'}`}>{recommendation?.label}</div> : null}
          </div>

          {analysis ? (
            <div className="signal-stack">
              <div className={`result-banner tone-${recommendation?.tone ?? 'neutral'}`}>
                <div className="result-banner-copy">
                  <div className="section-label">Decision band</div>
                  <h2>{recommendation?.label ?? 'Hold'}</h2>
                  <p>{recommendation?.summary}</p>
                </div>
                <div className="result-banner-metrics">
                  <div>
                    <span>Predicted alpha</span>
                    <strong>{analysis.predicted_alpha.toFixed(5)}</strong>
                  </div>
                  <div>
                    <span>Beta</span>
                    <strong>{analysis.beta.toFixed(3)}</strong>
                  </div>
                </div>
              </div>

              <div className="signal-grid">
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">Price</div>
                    <span className="metric-rank">Live</span>
                  </div>
                  <div className="metric-value">₹{analysis.current_price.toFixed(2)}</div>
                  <small>Current market price from the backend response.</small>
                </div>
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">Predicted return</div>
                    <span className="metric-rank">Model</span>
                  </div>
                  <div className="metric-value">{analysis.predicted_return.toFixed(5)}</div>
                  <small>Model-implied forward return for the selected ticker.</small>
                </div>
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">Alpha</div>
                    <span className="metric-rank">Edge</span>
                  </div>
                  <div className="metric-value">{analysis.predicted_alpha.toFixed(5)}</div>
                  <small>Adjusted edge after market expectation is removed.</small>
                </div>
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">Beta</div>
                    <span className="metric-rank">Risk</span>
                  </div>
                  <div className="metric-value">{analysis.beta.toFixed(3)}</div>
                  <small>Exposure relative to market movement.</small>
                </div>
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">Volatility</div>
                    <span className="metric-rank">Noise</span>
                  </div>
                  <div className="metric-value">{analysis.volatility.toFixed(5)}</div>
                  <small>Higher values indicate a more unsettled return path.</small>
                </div>
                <div className="metric">
                  <div className="metric-topline">
                    <div className="badge">RSI</div>
                    <span className="metric-rank">Momentum</span>
                  </div>
                  <div className="metric-value">{analysis.rsi.toFixed(1)}</div>
                  <small>Momentum framing from the feature pipeline.</small>
                </div>
              </div>

              <div className="summary-strip">
                <div className="summary-card">
                  <span>Model</span>
                  <strong>{analysis.model_name}</strong>
                  <small>Scored on the requested ticker.</small>
                </div>
                <div className="summary-card">
                  <span>Market expected return</span>
                  <strong>{analysis.market_expected_return.toFixed(5)}</strong>
                  <small>Reference used in the alpha calculation.</small>
                </div>
                <div className="summary-card">
                  <span>Interpretation</span>
                  <strong>{recommendation?.label ?? 'Hold'}</strong>
                  <small>{recommendation?.summary ?? 'Run inference to see a recommendation.'}</small>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="section-label">Waiting for input</div>
              <h3>Run inference to populate the signal cards.</h3>
              <p>The backend will return price, alpha, return, beta, volatility, RSI, and market context for the selected ticker.</p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}