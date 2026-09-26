import { ReactNode } from "react";

export function MetricCard({ label, value, helper, tone = "blue" }: { label: string; value: string; helper?: string; tone?: string }) {
  return <article className={`metric-card ${tone}`}><span className="metric-label">{label}</span><strong>{value}</strong>{helper && <small>{helper}</small>}</article>;
}
export function ChartCard({ title, subtitle, children, action }: { title: string; subtitle?: string; children: ReactNode; action?: ReactNode }) {
  return <section className="panel chart-card"><div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}
export function LoadingState({ label = "Loading analysis…" }: { label?: string }) { return <div className="state-box"><span className="loader" />{label}</div>; }
export function ErrorState({ message }: { message: string }) { return <div className="state-box error-state"><strong>Unable to load this view</strong><span>{message}</span></div>; }
export function CustomerSegmentBadge({ segment }: { segment: string }) { return <span className={`segment-badge ${segment.toLowerCase().split(" ").join("-")}`}>{segment}</span>; }
export function BudgetInput({ value, onChange }: { value: number; onChange: (value: number) => void }) { return <label className="field-label">Marketing budget<input type="number" min="0" value={value} onChange={(event) => onChange(Number(event.target.value))} /></label>; }
export function FileUpload({ onFile }: { onFile: (file: File) => void }) { return <label className="upload-zone"><span className="upload-icon">↥</span><strong>Drop campaign CSV here</strong><span>or click to browse · CSV up to 50 MB</span><input type="file" accept=".csv,text/csv" onChange={(event) => event.target.files?.[0] && onFile(event.target.files[0])} /></label>; }
export function DataTable({ columns, rows }: { columns: string[]; rows: Array<Record<string, unknown>> }) { return <div className="table-wrap"><table><thead><tr>{columns.map((column) => <th key={column}>{column.split("_").join(" ")}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={String(row.customer_id ?? index)}>{columns.map((column) => <td key={column}>{typeof row[column] === "number" ? Number(row[column]).toFixed(3) : String(row[column] ?? "—")}</td>)}</tr>)}</tbody></table></div>; }
