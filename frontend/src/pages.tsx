import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { api, CustomerResult, DataSummary, OptimizationResponse, SegmentResponse } from "./services/api";
import { BudgetInput, ChartCard, CustomerSegmentBadge, DataTable, ErrorState, FileUpload, LoadingState, MetricCard } from "./components/Primitives";

function useRequest<T>(loader: () => Promise<T>, enabled = true) {
  const [data, setData] = useState<T | null>(null); const [error, setError] = useState(""); const [loading, setLoading] = useState(enabled);
  useEffect(() => { if (!enabled) return; setLoading(true); loader().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); }, [enabled]); return { data, error, loading, setData };
}
const fmt = (value: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
type StageStatus = "Not Started" | "Running" | "Completed" | "Error";

function WorkflowStage({ index, label, status }: { index: number; label: string; status: StageStatus }) {
  return <div className={`workflow-step workflow-${status.toLowerCase().split(" ").join("-")}`}>
    <span>{status === "Completed" ? "✓" : index}</span>
    <div><strong>{label}</strong><small>{status}</small></div>
    <em>{status}</em>
  </div>;
}

export function Dashboard({ navigate }: { navigate: (page: string) => void }) {
  const summaryRequest = useRequest<DataSummary>(() => api.getSummary());
  const resultsRequest = useRequest<CustomerResult[]>(api.getResults, Boolean(summaryRequest.data?.has_data));
  const segmentsRequest = useRequest<SegmentResponse>(api.getSegments, Boolean(resultsRequest.data));
  const optimizationRequest = useRequest<OptimizationResponse>(api.getOptimizationResults, Boolean(resultsRequest.data));
  const auditRequest = useRequest<Record<string, unknown>>(api.getAudit, Boolean(resultsRequest.data));
  const data = summaryRequest.data;
  const results = resultsRequest.data ?? [];
  const modelReady = Boolean(resultsRequest.data);
  const anyLoading = summaryRequest.loading || (Boolean(data?.has_data) && resultsRequest.loading);
  if (summaryRequest.loading) return <LoadingState label="Loading dashboard data…" />;
  if (summaryRequest.error && !data) return <ErrorState message={summaryRequest.error} />;

  const avg = (field: keyof CustomerResult) => results.length ? results.reduce((total, row) => total + Number(row[field]), 0) / results.length : 0;
  const optimization = optimizationRequest.data?.summary;
  const stageStatus = (ready: boolean, loading: boolean, error: string, prerequisite: boolean): StageStatus =>
    loading ? "Running" : error ? "Error" : ready ? "Completed" : prerequisite ? "Not Started" : "Not Started";
  const datasetStatus: StageStatus = data?.has_data ? "Completed" : "Not Started";
  const modelStatus = stageStatus(modelReady, resultsRequest.loading, resultsRequest.error, Boolean(data?.has_data));
  const validationStatus = stageStatus(Boolean(auditRequest.data), auditRequest.loading, auditRequest.error, modelReady);
  const optimizationStatus = stageStatus(Boolean(optimization), optimizationRequest.loading, optimizationRequest.error, modelReady);
  const segmentStats = segmentsRequest.data?.statistics ?? [];
  const showDashboardError = resultsRequest.error && !modelReady && data?.has_data;
  if (showDashboardError && !results.length) {
    return <PageHeader eyebrow="WORKSPACE OVERVIEW" title="Good morning, marketing team" subtitle="Load data and train a causal model to activate the dashboard." action={<button className="primary-button" onClick={() => navigate("causal")}>Open causal analysis</button>}><div className="metric-grid"><MetricCard label="Total customers" value={fmt(data?.rows ?? 0)} helper="Current dataset" /><MetricCard label="Customers analyzed" value="—" helper="Model not trained" tone="purple" /></div><ErrorState message={resultsRequest.error} /></PageHeader>;
  }

  return <PageHeader eyebrow="WORKSPACE OVERVIEW" title="Good morning, marketing team" subtitle="Monitor causal performance and make the next promotion decision with confidence." action={<button className="primary-button" onClick={() => navigate("upload")}>＋ New campaign</button>}>
    <div className="metric-grid">
      <MetricCard label="Total customers" value={fmt(data?.rows ?? 0)} helper="Current dataset" />
      <MetricCard label="Customers analyzed" value={modelReady ? fmt(results.length) : "—"} helper={modelReady ? "Customer-level estimates" : "Train a model to begin"} tone="purple" />
      <MetricCard label="Average baseline probability" value={modelReady ? pct(avg("baseline_purchase_probability")) : "—"} helper="Estimated under $0" tone="green" />
      <MetricCard label="Average ITE for $10" value={modelReady ? pct(avg("ITE_10")) : "—"} helper="Incremental purchase probability" tone="amber" />
      <MetricCard label="Average ITE for $20" value={modelReady ? pct(avg("ITE_20")) : "—"} helper="Incremental purchase probability" tone="purple" />
      <MetricCard label="Persuadable customers" value={segmentStats.length ? fmt(segmentStats.find((item) => item.segment === "Persuadable")?.customer_count ?? 0) : "—"} helper="Server-classified response segment" tone="green" />
      <MetricCard label="Marketing budget" value={optimization ? `$${fmt(optimization.total_budget)}` : "—"} helper={optimization ? "Latest optimization run" : "Run optimization to set budget"} />
      <MetricCard label="Budget utilized" value={optimization ? `$${fmt(optimization.budget_used)}` : "—"} helper={optimization ? `${pct(optimization.total_budget ? optimization.budget_used / optimization.total_budget : 0)} of budget` : "No allocation yet"} tone="amber" />
      <MetricCard label="Expected incremental revenue" value={optimization ? `$${fmt(optimization.expected_incremental_revenue)}` : "—"} helper="Model-estimated campaign value" tone="green" />
    </div>
    <ChartCard title="Workflow status" subtitle="Dataset → Causal Model → Validation → Optimization">
      <div className="workflow-list workflow-status"><WorkflowStage index={1} label="Dataset" status={datasetStatus} /><WorkflowStage index={2} label="Causal Model" status={modelStatus} /><WorkflowStage index={3} label="Validation" status={validationStatus} /><WorkflowStage index={4} label="Optimization" status={optimizationStatus} /></div>
      {!modelReady && <div className="info-banner">No causal model is trained yet. Load data, then train the two discount models to populate customer-level charts.</div>}
      {anyLoading && <p className="muted">Refreshing downstream analysis status…</p>}
    </ChartCard>
    {modelReady ? <div className="dashboard-chart-grid">
      <ChartCard title="Customer treatment-effect distribution" subtitle="Estimated ITE across all analyzed customers">
        <Plot data={[{ x: results.map((row) => row.ITE_10), type: "histogram", name: "$10 ITE", marker: { color: "#3e78c3" }, opacity: 0.72 }, { x: results.map((row) => row.ITE_20), type: "histogram", name: "$20 ITE", marker: { color: "#8069ca" }, opacity: 0.62 }]} layout={{ ...plotLayout, barmode: "overlay", xaxis: { title: { text: "Incremental purchase probability" } }, yaxis: { title: { text: "Customers" } } }} config={plotConfig} style={{ width: "100%" }} />
      </ChartCard>
      <ChartCard title="Customer segment distribution" subtitle="Server-classified response groups">
        <Plot data={[{ x: segmentStats.map((item) => item.segment), y: segmentStats.map((item) => item.customer_count), type: "bar", marker: { color: ["#2baa78", "#3e78c3", "#d59b3e", "#c65f69", "#8a96aa"] } }]} layout={{ ...plotLayout, xaxis: { tickangle: -25 }, yaxis: { title: { text: "Customers" } } }} config={plotConfig} style={{ width: "100%" }} />
      </ChartCard>
      <ChartCard title="Baseline vs discounted purchase probability" subtitle="Average model-based potential outcomes">
        <Plot data={[{ x: ["Baseline ($0)", "$10 discount", "$20 discount"], y: [avg("baseline_purchase_probability"), avg("purchase_probability_10"), avg("purchase_probability_20")], type: "bar", marker: { color: ["#8a96aa", "#3e78c3", "#8069ca"] }, text: [avg("baseline_purchase_probability"), avg("purchase_probability_10"), avg("purchase_probability_20")].map(pct), textposition: "auto" }]} layout={{ ...plotLayout, yaxis: { title: { text: "Purchase probability" }, tickformat: ".0%" } }} config={plotConfig} style={{ width: "100%" }} />
      </ChartCard>
      <ChartCard title="ITE distribution" subtitle="Distribution summary for both discount levels">
        <Plot data={[{ y: results.map((row) => row.ITE_10), type: "box", name: "$10 ITE", marker: { color: "#3e78c3" }, boxmean: true }, { y: results.map((row) => row.ITE_20), type: "box", name: "$20 ITE", marker: { color: "#8069ca" }, boxmean: true }]} layout={{ ...plotLayout, yaxis: { title: { text: "Incremental purchase probability" } } }} config={plotConfig} style={{ width: "100%" }} />
      </ChartCard>
    </div> : <div className="empty-dashboard"><strong>Analysis charts will appear after training</strong><span>Customer effects, segments, and counterfactual probabilities are not available until the causal model completes.</span><button className="secondary-button" onClick={() => navigate("causal")}>Go to causal analysis</button></div>}
  </PageHeader>;
}

const plotLayout = { autosize: true, height: 280, margin: { l: 52, r: 16, t: 8, b: 58 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { size: 10, color: "#758197" }, legend: { orientation: "h" as const } };
const plotConfig = { displayModeBar: false, responsive: true };

function PageHeader({ eyebrow, title, subtitle, action, children }: { eyebrow: string; title: string; subtitle: string; action?: React.ReactNode; children: React.ReactNode }) { return <><div className="page-heading"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>{children}</>; }

export function UploadPage({ navigate }: { navigate: (page: string) => void }) {
  const [message, setMessage] = useState(""); const [error, setError] = useState(""); const [summary, setSummary] = useState<DataSummary | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null); const [loading, setLoading] = useState(false);
  const load = (promise: Promise<DataSummary>) => { setError(""); setMessage(""); setLoading(true); promise.then((result) => { setSummary(result); setMessage("Dataset validated successfully and is ready for causal analysis."); }).catch((e) => { setSummary(null); setError(userFriendlyUploadError(e.message)); }).finally(() => setLoading(false)); };
  const upload = (file: File) => { setError(""); setMessage(""); setSummary(null); parseCsvPreview(file).then(setPreview).catch(() => setPreview(null)); load(api.uploadData(file)); };
  const generate = () => { setPreview(null); load(api.generateData()); };
  return <PageHeader eyebrow="DATA MANAGEMENT" title="Load a campaign" subtitle="Validate your customer-level campaign data before training any causal model."><div className="upload-options"><button className="primary-button" onClick={generate} disabled={loading}>Generate Demo Dataset</button><span>or upload your own validated campaign</span></div><div className="upload-grid"><section className="panel"><h2>Upload CSV</h2><p className="muted">The backend validates schema, missing values, numeric fields, and treatment values before accepting the dataset.</p><FileUpload disabled={loading} onFile={upload} />{loading && <LoadingState label="Validating dataset…" />}{message && <div className="success-note">{message}</div>}{error && <ErrorState message={error} />}</section><section className="panel schema-panel"><h2>Expected schema</h2><div className="schema-list">{["customer_id", "age · income", "previous_purchases · website_visits", "customer_tenure · previous_spend", "engagement_score", "treatment · discount_amount", "purchase · revenue"].map((item) => <code key={item}>{item}</code>)}</div><div className="validation-note">Causal training remains unavailable until the dataset passes server validation.</div>{summary && <button className="primary-button" onClick={() => navigate("causal")}>Continue to causal analysis →</button>}</section></div>{preview ? <DatasetPreview preview={preview} summary={summary} /> : summary && <DatasetSummary summary={summary} />}</PageHeader>;
}

type CsvPreview = { columns: string[]; rows: Array<Record<string, string>>; rowCount: number; missing: Record<string, number> };
function parseCsvText(text: string): string[][] { return text.trim().split(/\r?\n/).filter(Boolean).map((line) => { const cells: string[] = []; let cell = ""; let quoted = false; for (let i = 0; i < line.length; i += 1) { const char = line[i]; if (char === '"') quoted = !quoted; else if (char === "," && !quoted) { cells.push(cell.trim()); cell = ""; } else cell += char; } cells.push(cell.trim()); return cells; }); }
function parseCsvPreview(file: File): Promise<CsvPreview> { return file.text().then((text) => { const parsed = parseCsvText(text); const columns = parsed[0] ?? []; const values = parsed.slice(1); const rows = values.slice(0, 5).map((cells) => Object.fromEntries(columns.map((column, index) => [column, cells[index] ?? ""]))); const missing = Object.fromEntries(columns.map((column, index) => [column, values.filter((cells) => !String(cells[index] ?? "").trim()).length])); return { columns, rows, rowCount: values.length, missing }; }); }
function userFriendlyUploadError(message: string) { if (message.toLowerCase().includes("missing required columns")) return `This CSV is missing required columns. Add the listed campaign fields and try again. ${message}`; if (message.toLowerCase().includes("missing values")) return `The CSV contains blank values. Fill in every required field before uploading. ${message}`; if (message.toLowerCase().includes("treatment")) return `Treatment values must be exactly 0, 10, or 20, and discount_amount must match treatment. ${message}`; if (message.toLowerCase().includes("numeric")) return `One or more numeric columns contain invalid text. Check the values and upload again. ${message}`; return message || "The dataset could not be validated. Check the CSV format and try again."; }
function DatasetPreview({ preview, summary }: { preview: CsvPreview; summary: DataSummary | null }) { const missingTotal = Object.values(preview.missing).reduce((total, count) => total + count, 0); return <section className="panel dataset-preview"><div className="panel-heading"><div><h2>Dataset preview</h2><p>First five rows from the selected CSV</p></div>{summary && <span className="pill green-pill">Validated</span>}</div><div className="preview-metrics"><MetricCard label="Rows" value={fmt(summary?.rows ?? preview.rowCount)} /><MetricCard label="Columns" value={fmt(summary?.columns.length ?? preview.columns.length)} /><MetricCard label="Missing values" value={fmt(missingTotal)} tone={missingTotal ? "amber" : "green"} /><MetricCard label="Purchase rate" value={summary ? pct(summary.purchase_rate) : "Pending"} tone="purple" /></div>{summary && <div className="treatment-summary"><strong>Treatment distribution</strong>{Object.entries(summary.treatment_counts).map(([treatment, count]) => <span key={treatment}>${treatment}: {fmt(count)} customers</span>)}</div>}{!summary && <div className="validation-note">Preview only. Final row counts, treatment distribution, and purchase rate are confirmed by the backend after validation.</div>}<DataTable columns={preview.columns} rows={preview.rows} /></section>; }
function DatasetSummary({ summary }: { summary: DataSummary }) { return <section className="panel dataset-preview"><div className="panel-heading"><div><h2>Dataset summary</h2><p>Server-validated campaign data</p></div><span className="pill green-pill">Validated</span></div><div className="preview-metrics"><MetricCard label="Rows" value={fmt(summary.rows)} /><MetricCard label="Columns" value={fmt(summary.columns.length)} /><MetricCard label="Missing values" value="0" tone="green" /><MetricCard label="Purchase rate" value={pct(summary.purchase_rate)} tone="purple" /></div><div className="treatment-summary"><strong>Treatment distribution</strong>{Object.entries(summary.treatment_counts).map(([treatment, count]) => <span key={treatment}>${treatment}: {fmt(count)} customers</span>)}</div></section>; }

export function CausalPage({ navigate }: { navigate: (page: string) => void }) {
  const [status, setStatus] = useState(""); const [error, setError] = useState(""); const run = () => { setStatus("Training…"); api.train().then(() => { setStatus("Models trained"); navigate("customers"); }).catch((e) => { setError(e.message); setStatus(""); }); };
  return <PageHeader eyebrow="CAUSAL ANALYSIS" title="Estimate incremental impact" subtitle="Double Machine Learning separates customers who would buy anyway from those persuaded by a discount." action={<button className="primary-button" onClick={run}>{status || "Train causal models"}</button>}><div className="model-grid"><ChartCard title="$10 discount vs $0" subtitle="Binary treatment contrast"><div className="model-card-copy"><strong>ITE₁₀</strong><span>Individual treatment effects</span><small>Observed confounders adjusted with cross-fitted nuisance models.</small></div></ChartCard><ChartCard title="$20 discount vs $0" subtitle="Binary treatment contrast"><div className="model-card-copy"><strong>ITE₂₀</strong><span>Individual treatment effects</span><small>Results preserve customer-level probability estimates.</small></div></ChartCard></div>{error && <ErrorState message={error} />}<div className="info-banner">Causal estimates are model-based effects, not guaranteed individual behavior. Review overlap and refutation diagnostics before acting.</div></PageHeader>;
}

export function CustomerPage({ navigate }: { navigate: (page: string) => void }) {
  const { data, loading, error } = useRequest<CustomerResult[]>(api.getResults);
  if (loading) return <LoadingState label="Loading customer effects…" />; if (error) return <ErrorState message={error} />;
  const rows = (data ?? []).slice(0, 25); return <PageHeader eyebrow="CUSTOMER INSIGHTS" title="Who should receive an offer?" subtitle="Explore customer-level counterfactual purchase probabilities and incremental effects." action={<button className="secondary-button" onClick={() => navigate("optimization")}>Open budget optimizer →</button>}><div className="metric-grid"><MetricCard label="Customers scored" value={fmt(data?.length ?? 0)} helper="Customer-level outputs" /><MetricCard label="Average ITE $10" value={`${((data?.reduce((sum, row) => sum + row.ITE_10, 0) ?? 0) / Math.max(1, data?.length ?? 1) * 100).toFixed(1)} pp`} helper="Estimated increment" tone="green" /><MetricCard label="Average ITE $20" value={`${((data?.reduce((sum, row) => sum + row.ITE_20, 0) ?? 0) / Math.max(1, data?.length ?? 1) * 100).toFixed(1)} pp`} helper="Estimated increment" tone="purple" /></div><ChartCard title="Incremental effect distribution" subtitle="First 25 scored customers"><Plot data={[{ x: rows.map((row) => row.customer_id), y: rows.map((row) => row.ITE_10), type: "bar", name: "$10 ITE", marker: { color: "#3e78c3" } }, { x: rows.map((row) => row.customer_id), y: rows.map((row) => row.ITE_20), type: "bar", name: "$20 ITE", marker: { color: "#8069ca" } }]} layout={{ autosize: true, height: 280, margin: { l: 42, r: 16, t: 10, b: 48 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { size: 10, color: "#758197" }, barmode: "group", legend: { orientation: "h" } }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%" }} /></ChartCard><ChartCard title="Customer effect table" subtitle="Showing the first 25 customers"><DataTable columns={["customer_id", "baseline_purchase_probability", "purchase_probability_10", "purchase_probability_20", "ITE_10", "ITE_20"]} rows={rows as unknown as Array<Record<string, unknown>>} /></ChartCard></PageHeader>;
}

export function OptimizationPage() {
  const [budget, setBudget] = useState(10000); const [result, setResult] = useState<OptimizationResponse | null>(null); const [error, setError] = useState(""); const run = () => api.runOptimization(budget).then(setResult).catch((e) => setError(e.message));
  return <PageHeader eyebrow="BUDGET OPTIMIZATION" title="Allocate every dollar deliberately" subtitle="Choose the best discrete action for each customer while staying within your marketing budget." action={<button className="primary-button" onClick={run}>Run optimization</button>}><div className="optimization-controls panel"><BudgetInput value={budget} onChange={setBudget} /><div className="constraint-note">Actions: $0, $10, $20 · one action per customer</div></div>{error && <ErrorState message={error} />}{result && <><div className="metric-grid"><MetricCard label="Budget used" value={`$${fmt(result.summary.budget_used)}`} helper={`$${fmt(result.summary.remaining_budget)} remaining`} /><MetricCard label="Expected incremental value" value={`$${fmt(result.summary.expected_incremental_revenue)}`} helper="Model-estimated" tone="green" /><MetricCard label="Customers targeted" value={fmt(result.summary.number_targeted)} helper={`${result.summary.number_no_discount} no discount`} tone="purple" /></div><ChartCard title="Recommended allocations"><DataTable columns={["customer_id", "recommended_discount", "expected_purchase_probability", "expected_incremental_revenue", "discount_cost", "estimated_net_value"]} rows={result.recommendations} /></ChartCard></>}</PageHeader>;
}

export function AuditPage() {
  const { data, loading, error } = useRequest<Record<string, unknown>>(api.getAudit);
  if (loading) return <LoadingState label="Running causal audit…" />; if (error) return <ErrorState message={error} />;
  const audits = Object.values(data ?? {}) as Array<{ discount_amount: number; results: Array<Record<string, unknown>> }>; return <PageHeader eyebrow="CAUSAL AUDIT" title="Stress-test the estimate" subtitle="Refutation diagnostics challenge assumptions; they do not prove the model is correct."><div className="audit-grid">{audits.map((audit) => <section className="panel" key={audit.discount_amount}><div className="panel-heading"><div><h2>${audit.discount_amount} vs $0</h2><p>DoWhy sensitivity checks</p></div><span className="audit-count">{audit.results.length} checks</span></div>{audit.results.map((item) => <div className="audit-row" key={String(item.test_name)}><div><strong>{String(item.test_name)}</strong><small>{String(item.interpretation)}</small></div><span className={item.status === "completed" ? "pill green-pill" : "pill amber-pill"}>{String(item.status)}</span></div>)}</section>)}</div></PageHeader>;
}
