import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { api, CustomerResult, CausalModelSummary, DataSummary, OptimizationResponse, SegmentResponse, UpliftEvaluation, UpliftCurve } from "./services/api";
import { BudgetInput, ChartCard, CustomerSegmentBadge, DataTable, ErrorState, FileUpload, LoadingState, MetricCard } from "./components/Primitives";

function useRequest<T>(loader: () => Promise<T>, enabled = true, key?: string | number) {
  const [data, setData] = useState<T | null>(null); const [error, setError] = useState(""); const [loading, setLoading] = useState(enabled);
  useEffect(() => { if (!enabled) return; setLoading(true); setError(""); loader().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); }, [enabled, key]); return { data, error, loading, setData };
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
  const [status, setStatus] = useState(""); const [error, setError] = useState(""); const [refresh, setRefresh] = useState(0);
  const run = () => { setStatus("Training…"); setError(""); api.train().then(() => { setStatus("Models trained"); setRefresh((value) => value + 1); }).catch((e) => { setError(e.message); setStatus(""); }); };
  const summaryRequest = useRequest<CausalModelSummary>(() => api.getCausalSummary(), true, refresh);
  const resultsRequest = useRequest<CustomerResult[]>(() => api.getResults(), true, refresh);
  const segmentsRequest = useRequest<SegmentResponse>(() => api.getSegments(), true, refresh);
  const upliftRequest = useRequest<Record<string, UpliftEvaluation>>(api.getUplift, Boolean(resultsRequest.data), refresh);
  if (summaryRequest.loading || resultsRequest.loading) return <LoadingState label="Loading causal analysis…" />;
  const notTrained = summaryRequest.error || resultsRequest.error;
  if (notTrained && !summaryRequest.data) return <PageHeader eyebrow="CAUSAL ANALYSIS" title="Estimate incremental impact" subtitle="Train the two binary discount models to unlock treatment effects and counterfactual probabilities." action={<button className="primary-button" onClick={run} disabled={status === "Training…"}>{status || "Train causal models"}</button>}><div className="empty-dashboard"><strong>Causal model not trained</strong><span>{error || "No customer-level results are available yet. Load a validated dataset before training."}</span></div>{error && <ErrorState message={error} />}</PageHeader>;
  const rows = resultsRequest.data ?? [];
  const models = summaryRequest.data?.models ?? {};
  return <PageHeader eyebrow="CAUSAL ANALYSIS" title="Estimate incremental impact" subtitle="Double Machine Learning separates customers who would buy anyway from those persuaded by a discount." action={<button className="primary-button" onClick={run} disabled={status === "Training…"}>{status || "Retrain causal models"}</button>}>
    <ChartCard title="Causal model status" subtitle="Separate binary contrasts trained against the no-discount control">
      <div className="model-status-grid">{[["10", "$10 discount vs $0"], ["20", "$20 discount vs $0"]].map(([discount, label]) => { const model = models[discount]; return <div className="model-status-card" key={discount}><div><strong>{label}</strong><span className="pill green-pill">Completed</span></div><small>{model ? `${fmt(model.treatment_count)} treated · ${fmt(model.control_count)} control observations` : "Model metadata unavailable"}</small></div>; })}</div>
      {status && <div className="success-note">{status}</div>}{error && <ErrorState message={error} />}
    </ChartCard>
    <ChartCard title="Causal DAG explanation" subtitle="Observed covariates are used to adjust treatment assignment and purchase outcomes">
      <div className="dag-flow"><div className="dag-node"><strong>Customer features</strong><span>Age · income · purchases · visits · tenure · spend · engagement</span></div><b>→</b><div className="dag-node"><strong>Treatment</strong><span>$0, $10, or $20 discount</span></div><b>→</b><div className="dag-node"><strong>Purchase</strong><span>Binary purchase outcome</span></div></div><p className="muted">The model estimates effects for $10 vs $0 and $20 vs $0 under the assumption that important confounders are observed and overlap is adequate. Estimates are not guaranteed individual behavior.</p>
    </ChartCard>
    <div className="metric-grid"><MetricCard label="ATE for $10" value={models["10"] ? pct(models["10"].ate) : "—"} helper={intervalLabel(models["10"])} tone="green" /><MetricCard label="ATE for $20" value={models["20"] ? pct(models["20"].ate) : "—"} helper={intervalLabel(models["20"])} tone="purple" /></div>
    <div className="dashboard-chart-grid"><ChartCard title="Customer ITE distribution" subtitle="Individual treatment effects across analyzed customers"><Plot data={[{ x: rows.map((row) => row.ITE_10), type: "histogram", name: "$10 ITE", marker: { color: "#3e78c3" }, opacity: 0.72 }, { x: rows.map((row) => row.ITE_20), type: "histogram", name: "$20 ITE", marker: { color: "#8069ca" }, opacity: 0.62 }]} layout={{ ...plotLayout, barmode: "overlay", xaxis: { title: { text: "ITE" } }, yaxis: { title: { text: "Customers" } } }} config={plotConfig} style={{ width: "100%" }} /></ChartCard><ChartCard title="Counterfactual probabilities" subtitle="Average model-based potential outcomes"><Plot data={[{ x: ["Baseline ($0)", "$10", "$20"], y: [average(rows, "baseline_purchase_probability"), average(rows, "purchase_probability_10"), average(rows, "purchase_probability_20")], type: "bar", marker: { color: ["#8a96aa", "#3e78c3", "#8069ca"] }, text: [average(rows, "baseline_purchase_probability"), average(rows, "purchase_probability_10"), average(rows, "purchase_probability_20")].map(pct), textposition: "auto" }]} layout={{ ...plotLayout, yaxis: { title: { text: "Probability" }, tickformat: ".0%" } }} config={plotConfig} style={{ width: "100%" }} /></ChartCard></div>
    <CustomerAnalysisTable rows={rows} segments={segmentsRequest.data?.customers ?? []} />
    <UpliftAnalysis evaluations={upliftRequest.data} loading={upliftRequest.loading} error={upliftRequest.error} />
  </PageHeader>;
}

function average(rows: CustomerResult[], field: keyof CustomerResult) { return rows.length ? rows.reduce((total, row) => total + Number(row[field]), 0) / rows.length : 0; }
function intervalLabel(model?: CausalModelSummary["models"][string]) { return model?.ate_lower !== null && model?.ate_lower !== undefined && model?.ate_upper !== null && model?.ate_upper !== undefined ? `95% CI ${pct(model.ate_lower)} to ${pct(model.ate_upper)}` : "Estimated average treatment effect"; }
function CustomerAnalysisTable({ rows, segments }: { rows: CustomerResult[]; segments: Array<CustomerResult & { segment: string }> }) {
  const [query, setQuery] = useState(""); const [sort, setSort] = useState<keyof CustomerResult>("customer_id"); const [ascending, setAscending] = useState(true); const [page, setPage] = useState(1); const pageSize = 15;
  const segmentById = new Map(segments.map((row) => [row.customer_id, row.segment]));
  const filtered = rows.filter((row) => row.customer_id.toLowerCase().includes(query.toLowerCase()));
  const sorted = [...filtered].sort((left, right) => { const a = String(left[sort]); const b = String(right[sort]); const numeric = sort !== "customer_id"; const result = numeric ? Number(a) - Number(b) : a.localeCompare(b); return ascending ? result : -result; });
  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize)); const visible = sorted.slice((page - 1) * pageSize, page * pageSize);
  const changeSort = (column: keyof CustomerResult) => { if (sort === column) setAscending((value) => !value); else { setSort(column); setAscending(true); } setPage(1); };
  const headers: Array<[keyof CustomerResult, string]> = [["customer_id", "Customer ID"], ["baseline_purchase_probability", "Baseline Probability"], ["purchase_probability_10", "$10 Probability"], ["purchase_probability_20", "$20 Probability"], ["ITE_10", "ITE $10"], ["ITE_20", "ITE $20"]];
  return <ChartCard title="Customer-level treatment effects" subtitle={`${fmt(filtered.length)} matching customers · page ${page} of ${pageCount}`}><div className="table-toolbar"><input aria-label="Search customers" placeholder="Search customer ID" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /><span>{fmt(rows.length)} analyzed</span></div><div className="table-wrap"><table><thead><tr>{headers.map(([column, label]) => <th key={column}><button className="table-sort" onClick={() => changeSort(column)}>{label} {sort === column ? (ascending ? "↑" : "↓") : ""}</button></th>)}<th>Segment</th></tr></thead><tbody>{visible.map((row) => <tr key={row.customer_id}><td>{row.customer_id}</td><td>{pct(row.baseline_purchase_probability)}</td><td>{pct(row.purchase_probability_10)}</td><td>{pct(row.purchase_probability_20)}</td><td>{pct(row.ITE_10)}</td><td>{pct(row.ITE_20)}</td><td>{segmentById.get(row.customer_id) ?? "—"}</td></tr>)}</tbody></table></div><div className="pagination"><button className="secondary-button" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Previous</button><span>Page {page} of {pageCount}</span><button className="secondary-button" disabled={page >= pageCount} onClick={() => setPage((value) => value + 1)}>Next</button></div></ChartCard>;
}

function UpliftAnalysis({ evaluations, loading, error }: { evaluations: Record<string, UpliftEvaluation> | null; loading: boolean; error: string }) {
  if (loading) return <ChartCard title="Uplift analysis" subtitle="Loading backend evaluation curves…"><LoadingState label="Loading uplift evaluation…" /></ChartCard>;
  if (error) return <ChartCard title="Uplift analysis" subtitle="Backend evaluation diagnostics"><ErrorState message={error} /></ChartCard>;
  if (!evaluations || (!evaluations["10"] && !evaluations["20"])) return <ChartCard title="Uplift analysis" subtitle="Available after causal training"><div className="empty-dashboard"><strong>No uplift evaluation available</strong><span>Train the causal models to calculate ranked uplift and Qini diagnostics.</span></div></ChartCard>;
  return <section className="uplift-section"><div className="section-heading"><div><span className="eyebrow">UPLIFT EVALUATION</span><h2>Targeting diagnostics</h2><p>Ranked cumulative gains compare targeting by estimated ITE with a random-targeting baseline.</p></div></div><div className="uplift-grid">{[["10", "Uplift curve", "uplift", "Ranks customers by estimated $10 incremental effect."], ["20", "Uplift curve", "uplift", "Ranks customers by estimated $20 incremental effect."], ["10", "Qini curve", "qini", "Shows the Qini-style cumulative incremental gain for $10 vs $0."], ["20", "Qini curve", "qini", "Shows the Qini-style cumulative incremental gain for $20 vs $0."]].map(([discount, title, curveName, explanation]) => { const evaluation = evaluations[discount]; const curve = evaluation?.curves[curveName as "uplift" | "qini"]; return curve ? <UpliftChart key={`${discount}-${curveName}`} discount={Number(discount)} title={`${title}: $${discount} vs $0`} explanation={explanation} curve={curve} /> : null; })}</div><div className="info-banner">These curves are observational diagnostics based on estimated effects and observed outcomes. A higher curve does not universally prove causal superiority; review overlap, refutation results, and campaign context before acting.</div></section>;
}

function UpliftChart({ discount, title, explanation, curve }: { discount: number; title: string; explanation: string; curve: UpliftCurve }) {
  const population = curve.x.map((rank) => rank / Math.max(1, curve.x.length));
  const customData = curve.x.map((rank, index) => [rank, population[index], curve.y[index], `$${discount}`]);
  return <ChartCard title={title} subtitle={`${curve.treated_count} treated · ${curve.control_count} control`}><Plot data={[{ x: population, y: curve.y, type: "scatter", mode: "lines", name: "Estimated gain", line: { color: discount === 10 ? "#3e78c3" : "#8069ca", width: 3 }, customdata: customData, hovertemplate: "Rank: %{customdata[0]}<br>Cumulative population: %{customdata[1]:.1%}<br>Incremental gain: %{customdata[2]:.3f}<br>Treatment level: %{customdata[3]}<extra></extra>" }, { x: population, y: curve.random_baseline, type: "scatter", mode: "lines", name: "Random targeting baseline", line: { color: "#a4adba", dash: "dash" }, hovertemplate: "Rank: %{x:.0%}<br>Random baseline gain: %{y:.3f}<extra></extra>" }]} layout={{ ...plotLayout, xaxis: { title: { text: "Cumulative population targeted" }, tickformat: ".0%" }, yaxis: { title: { text: "Cumulative incremental gain" } } }} config={plotConfig} style={{ width: "100%" }} /><p className="chart-explanation">{explanation} The shaded interpretation should be cautious: this is not proof that one policy dominates for every population.</p><div className="curve-statistics"><span>Area under curve <strong>{curve.area_under_curve.toFixed(2)}</strong></span><span>Final incremental gain <strong>{curve.final_incremental_gain.toFixed(2)}</strong></span></div></ChartCard>;
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
