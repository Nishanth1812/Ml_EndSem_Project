"use client";

import { useEffect, useState } from 'react';

import type { Metric } from '../../lib/types';

type MetricsData = {
  metrics: Record<string, Metric>;
  timestamp?: string;
  description?: string;
};

type MetricCardProps = {
  label: string;
  metric: Metric;
};

function MetricCard({ label, metric }: MetricCardProps) {
  return (
    <div className="metric">
      <div className="metric-topline">
        <div className="badge">{label}</div>
        <span className="metric-rank">Test</span>
      </div>
      <div className="metric-value">R² {metric.r2.toFixed(4)}</div>
      <small>MAE {metric.mae.toFixed(6)} · RMSE {metric.rmse.toFixed(6)}</small>
    </div>
  );
}

export default function Page() {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch('/metrics.json');
        if (!response.ok) {
          throw new Error('Failed to load metrics');
        }
        const data = await response.json() as MetricsData;
        setMetricsData(data);
      } catch (exception) {
        setError(exception instanceof Error ? exception.message : 'Unable to load metrics');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const metrics = metricsData?.metrics ?? {};
  const metricEntries = Object.entries(metrics);
  const bestModel = [...metricEntries].sort((left, right) => right[1].r2 - left[1].r2)[0];

  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Model Metrics</h1>
          <p className="subtitle">
            Performance metrics for all trained models on the holdout test set. Metrics are loaded from the metrics configuration file.
          </p>
        </div>

        <div className="hero-aside hero-summary-grid">
          <div className="status">
            <strong>Best model</strong>
            <span>{bestModel ? bestModel[0].replace(/_/g, ' ') : '—'}</span>
          </div>
          <div className="status">
            <strong>Models</strong>
            <span>{metricEntries.length > 0 ? `${metricEntries.length} models` : '—'}</span>
          </div>
        </div>
      </section>

      {error ? <div className="status warn">{error}</div> : null}
      {loading ? <div className="status">Loading metrics...</div> : null}

      {!loading && metricEntries.length > 0 ? (
        <section className="grid two">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <div className="section-label">Best performer</div>
                <h2>Top model</h2>
              </div>
            </div>

            {bestModel ? (
              <div className="summary-strip">
                <div className="summary-card">
                  <span>Model</span>
                  <strong>{bestModel[0].replace(/_/g, ' ')}</strong>
                  <small>Highest R² score</small>
                </div>
                <div className="summary-card">
                  <span>R² Score</span>
                  <strong>{bestModel[1].r2.toFixed(4)}</strong>
                  <small>Variance explained</small>
                </div>
                <div className="summary-card">
                  <span>MAE</span>
                  <strong>{bestModel[1].mae.toFixed(6)}</strong>
                  <small>Mean absolute error</small>
                </div>
                <div className="summary-card">
                  <span>RMSE</span>
                  <strong>{bestModel[1].rmse.toFixed(6)}</strong>
                  <small>Root mean square error</small>
                </div>
              </div>
            ) : null}
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <div className="section-label">Summary</div>
                <h2>All models</h2>
              </div>
            </div>

            <div className="summary-strip">
              <div className="summary-card">
                <span>Total models</span>
                <strong>{metricEntries.length}</strong>
                <small>Evaluated models</small>
              </div>
              <div className="summary-card">
                <span>Average R²</span>
                <strong>
                  {(
                    metricEntries.reduce((sum, [, m]) => sum + m.r2, 0) /
                    metricEntries.length
                  ).toFixed(4)}
                </strong>
                <small>Mean performance</small>
              </div>
              {metricsData?.timestamp ? (
                <div className="summary-card">
                  <span>Updated</span>
                  <strong>{new Date(metricsData.timestamp).toLocaleDateString()}</strong>
                  <small>Metrics timestamp</small>
                </div>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      {!loading && metricEntries.length > 0 ? (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Performance</div>
              <h2>Detailed metrics</h2>
            </div>
            <div className="panel-note">All models</div>
          </div>

          <div className="metric-grid">
            {metricEntries.map(([name, metric]) => (
              <MetricCard key={name} label={name.replace(/_/g, ' ')} metric={metric} />
            ))}
          </div>
        </section>
      ) : null}

      {!loading && metricEntries.length === 0 ? (
        <div className="empty-state">
          <div className="section-label">No metrics found</div>
          <h3>Metrics file not found or empty</h3>
          <p>Place a metrics.json file in the public folder to display model performance data.</p>
        </div>
      ) : null}
    </main>
  );
}
