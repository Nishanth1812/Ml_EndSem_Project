"use client";

import { useEffect, useState } from 'react';

import { fetchJson } from '../../lib/api';
import type { Metric, MetricsResponse, StatusResponse, TrainResponse } from '../../lib/types';

type MetricCardProps = {
  label: string;
  metric: Metric;
};

function MetricCard({ label, metric }: MetricCardProps) {
  return (
    <div className="metric">
      <div className="metric-topline">
        <div className="badge">{label}</div>
        <span className="metric-rank">Holdout</span>
      </div>
      <div className="metric-value">R2 {metric.r2.toFixed(4)}</div>
      <small>MAE {metric.mae.toFixed(6)} · RMSE {metric.rmse.toFixed(6)}</small>
    </div>
  );
}

export default function Page() {
  const [health, setHealth] = useState<StatusResponse | null>(null);
  const [metrics, setMetrics] = useState<Record<string, Metric> | null>(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function loadMetrics() {
    const status = await fetchJson<StatusResponse>('/health');
    setHealth(status);

    if (status.trained) {
      const payload = await fetchJson<MetricsResponse>('/metrics');
      setMetrics(payload.metrics);
    } else {
      setMetrics(null);
    }
  }

  useEffect(() => {
    void (async () => {
      try {
        await loadMetrics();
      } catch (exception) {
        setError(exception instanceof Error ? exception.message : 'Unable to load training status');
      }
    })();
  }, []);

  async function handleTrain() {
    setLoading(true);
    setMessage('');
    setError('');
    try {
      const payload = await fetchJson<TrainResponse>('/train', { method: 'POST' });
      setMessage(payload.message);
      setMetrics(payload.metrics);
      await loadMetrics();
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : 'Training failed');
    } finally {
      setLoading(false);
    }
  }

  const metricEntries = metrics ? Object.entries(metrics) : [];
  const bestModel = [...metricEntries].sort((left, right) => right[1].r2 - left[1].r2)[0];

  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Model Training</h1>
          <p className="subtitle">
            Run training to fit all models on historical data and review performance metrics on the holdout test set.
          </p>
        </div>

        <div className="hero-aside hero-summary-grid">
          <div className={`status ${health?.trained ? 'ok' : 'warn'}`}>
            <strong>Status</strong>
            <span>{health?.trained ? 'Ready' : 'Untrained'}</span>
          </div>
          <div className="status">
            <strong>Best model</strong>
            <span>{bestModel ? bestModel[0].replace(/_/g, ' ') : '—'}</span>
          </div>
        </div>
      </section>

      {error ? <div className="status warn">{error}</div> : null}
      {message ? <div className="status ok">{message}</div> : null}

      <section className="grid two">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Training flow</div>
              <h2>Chronological fit</h2>
            </div>
            <div className="panel-note">API /train</div>
          </div>

          <div className="stack-list">
            <div className="stack-item">
              <strong>1. Load and prepare data</strong>
              <small>Feature engineering and target creation happen inside the backend engine.</small>
            </div>
            <div className="stack-item">
              <strong>2. Fit baseline and tree models</strong>
              <small>Linear Regression is the benchmark, while Random Forest and XGBoost are the inference models.</small>
            </div>
            <div className="stack-item">
              <strong>3. Review holdout performance</strong>
              <small>Metrics are returned from the backend for the trained model stack.</small>
            </div>
          </div>

          <div className="actions" style={{ marginTop: 18 }}>
            <button className="btn btn-primary" onClick={handleTrain} type="button" disabled={loading}>
              {loading ? 'Training...' : 'Train models'}
            </button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Evaluation</div>
              <h2>Holdout metrics</h2>
            </div>
            <div className="panel-note">Sorted by R2</div>
          </div>

          <div className="summary-strip">
            <div className="summary-card">
              <span>Best model</span>
              <strong>{bestModel ? bestModel[0].replace('_', ' ') : '—'}</strong>
              <small>{bestModel ? `R2 ${bestModel[1].r2.toFixed(4)}` : 'Train models to see the best performer.'}</small>
            </div>
            <div className="summary-card">
              <span>Mode</span>
              <strong>{health?.trained ? 'Ready for inference' : 'Not trained'}</strong>
              <small>Inference routes read from the same backend service.</small>
            </div>
          </div>

          <div className="metric-grid" style={{ marginTop: 16 }}>
            {metricEntries.length > 0 ? (
              metricEntries.map(([name, metric]) => <MetricCard key={name} label={name.replace('_', ' ')} metric={metric} />)
            ) : (
              <div className="empty-state">
                <div className="section-label">No metrics yet</div>
                <h3>Train models to populate the evaluation view.</h3>
                <p>The backend will return MAE, RMSE, and R2 for the baseline, Random Forest, and XGBoost models.</p>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
