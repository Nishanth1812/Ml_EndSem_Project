"use client";

import { useEffect, useState } from 'react';

import { fetchJson } from '../../lib/api';
import type { DataResponse, StatusResponse } from '../../lib/types';

export default function Page() {
  const [health, setHealth] = useState<StatusResponse | null>(null);
  const [data, setData] = useState<DataResponse | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      try {
        const status = await fetchJson<StatusResponse>('/health');
        const payload = await fetchJson<DataResponse>('/data');
        setHealth(status);
        setData(payload);
      } catch (exception) {
        setError(exception instanceof Error ? exception.message : 'Unable to load market data');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <main className="page-stack">
      <section className="hero dark-hero hero-grid">
        <div className="hero-copy">
          <h1>Historical Data</h1>
          <p className="subtitle">
            View the market data, available columns, and recent price history used by the models.
          </p>
        </div>

        <div className="hero-aside hero-summary-grid">
          <div className="status">
            <strong>Rows</strong>
            <span>{data?.rows ? data.rows.toLocaleString() : '—'}</span>
          </div>
          <div className="status">
            <strong>Columns</strong>
            <span>{data?.columns?.length ? `${data.columns.length} fields` : '—'}</span>
          </div>
        </div>
      </section>

      {error ? <div className="status warn">{error}</div> : null}

      <section className="grid two">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Dataset snapshot</div>
              <h2>Structure</h2>
            </div>
            <div className="panel-note">Source: FastAPI /data</div>
          </div>

          <div className="summary-strip">
            <div className="summary-card">
              <span>Rows</span>
              <strong>{data ? data.rows.toLocaleString() : '—'}</strong>
              <small>Cleaned history returned by the API.</small>
            </div>
            <div className="summary-card">
              <span>Columns</span>
              <strong>{data?.columns?.length ? data.columns.length : '—'}</strong>
              <small>Available fields in the loaded frame.</small>
            </div>
            <div className="summary-card">
              <span>Preview rows</span>
              <strong>{data?.preview?.length ?? 0}</strong>
              <small>Latest rows exposed by the backend.</small>
            </div>
          </div>

          {data ? (
            <div className="stack-list" style={{ marginTop: 16 }}>
              <div className="stack-item">
                <strong>Columns</strong>
                <small>{data.columns.join(', ')}</small>
              </div>
            </div>
          ) : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <div className="section-label">Preview</div>
              <h2>Latest sample</h2>
            </div>
            <div className="panel-note">Tail rows</div>
          </div>

          {data?.preview?.length ? (
            <div className="table-shell">
              <table>
                <thead>
                  <tr>
                    {Object.keys(data.preview[0] ?? {}).map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.preview.map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, valueIndex) => (
                        <td key={valueIndex}>{String(value ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="section-label">No preview yet</div>
              <h3>Load the dataset to see the latest rows.</h3>
              <p>The backend will return the cleaned market dataset and the last five rows when ready.</p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
