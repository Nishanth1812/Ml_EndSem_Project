export type StatusResponse = {
  status: string;
  data_path: string;
  trained: boolean;
};

export type Metric = {
  mae: number;
  rmse: number;
  r2: number;
};

export type MetricsResponse = {
  metrics: Record<string, Metric>;
};

export type DataResponse = {
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
};

export type TrainResponse = {
  message: string;
  metrics: Record<string, Metric>;
};

export type AnalysisResponse = {
  ticker: string;
  model_name: string;
  current_price: number;
  predicted_return: number;
  market_expected_return: number;
  predicted_alpha: number;
  beta: number;
  volatility: number;
  rsi: number;
};

export type RankResponse = {
  items: AnalysisResponse[];
};

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const MODEL_OPTIONS = [
  { value: 'baseline', label: 'Linear Regression' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'xgboost', label: 'XGBoost' },
] as const;

export const NAV_ITEMS = [
  { href: '/', label: 'Overview' },
  { href: '/data', label: 'Data' },
  { href: '/metrics', label: 'Metrics' },
  { href: '/inference', label: 'Inference' },
  { href: '/screener', label: 'Screener' },
] as const;
