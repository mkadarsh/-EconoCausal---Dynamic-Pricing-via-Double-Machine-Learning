const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

export interface DataSummary {
  rows: number;
  columns: string[];
  treatment_counts: Record<string, number>;
  purchase_rate: number;
  has_data: boolean;
}

export interface CustomerResult {
  customer_id: string;
  baseline_purchase_probability: number;
  purchase_probability_10: number;
  purchase_probability_20: number;
  ITE_10: number;
  ITE_20: number;
}

export interface SegmentStatistic {
  segment: string;
  customer_count: number;
  customer_fraction: number;
}

export interface SegmentResponse {
  customers: Array<CustomerResult & { segment: string }>;
  statistics: SegmentStatistic[];
}

export interface UpliftCurve {
  discount_amount: number;
  x: number[];
  y: number[];
  random_baseline: number[];
  metric: string;
  area_under_curve: number;
  final_incremental_gain: number;
}

export interface OptimizationSummary {
  total_budget: number;
  budget_used: number;
  remaining_budget: number;
  expected_incremental_revenue: number;
  expected_discount_cost: number;
  number_targeted: number;
  number_no_discount: number;
}

export interface OptimizationResponse {
  recommendations: Array<Record<string, string | number>>;
  summary: OptimizationSummary;
}

export interface Api {
  getSummary(): Promise<DataSummary>;
  generateData(n_customers?: number): Promise<DataSummary>;
  uploadData(file: File): Promise<DataSummary>;
  train(): Promise<{ status: string; models: number[]; rows: number; columns: string[] }>;
  getResults(): Promise<CustomerResult[]>;
  getSegments(): Promise<SegmentResponse>;
  getCustomer(id: string): Promise<CustomerResult>;
  getUplift(): Promise<Record<string, UpliftCurve>>;
  getPropensity(): Promise<Record<string, unknown>>;
  getAudit(): Promise<Record<string, unknown>>;
  runOptimization(budget: number, expectedOrderValue?: number): Promise<OptimizationResponse>;
  getOptimizationResults(): Promise<OptimizationResponse>;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api: Api = {
  getSummary: () => request<DataSummary>("/data/summary"),
  generateData: (n_customers = 5000) =>
    request<DataSummary>("/data/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_customers, random_seed: 42 }),
    }),
  uploadData: (file) => {
    const body = new FormData();
    body.append("file", file);
    return request<DataSummary>("/data/upload", { method: "POST", body });
  },
  train: () => request("/causal/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  getResults: () => request<CustomerResult[]>("/causal/results"),
  getSegments: () => request<SegmentResponse>("/causal/segments"),
  getCustomer: (id) => request<CustomerResult>(`/causal/customer/${encodeURIComponent(id)}`),
  getUplift: () => request<Record<string, UpliftCurve>>("/causal/uplift"),
  getPropensity: () => request<Record<string, unknown>>("/causal/propensity"),
  getAudit: () => request<Record<string, unknown>>("/causal/audit"),
  runOptimization: (budget, expectedOrderValue = 100) =>
    request<OptimizationResponse>("/optimization/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ budget, expected_order_value: expectedOrderValue }),
    }),
  getOptimizationResults: () => request<OptimizationResponse>("/optimization/results"),
};
