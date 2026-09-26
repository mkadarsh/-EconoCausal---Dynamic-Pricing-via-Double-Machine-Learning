import { ReactNode } from "react";

export const navItems = [
  ["dashboard", "Overview", "⌂"],
  ["upload", "Data upload", "↥"],
  ["causal", "Causal analysis", "◉"],
  ["customers", "Customer insights", "◌"],
  ["optimization", "Budget optimization", "◈"],
  ["audit", "Causal audit", "✓"],
] as const;

export function Header({ onMenu }: { onMenu: () => void }) {
  return <header className="topbar"><button className="mobile-menu" onClick={onMenu}>☰</button><div className="brand-mark">E</div><div><strong>EconoCausal</strong><span>Causal pricing intelligence</span></div><div className="topbar-right"><span className="status-dot" /> API connected</div></header>;
}

export function Sidebar({ page, navigate, open }: { page: string; navigate: (page: string) => void; open: boolean }) {
  return <aside className={`sidebar ${open ? "open" : ""}`}><div className="side-label">WORKSPACE</div>{navItems.map(([key, label, icon]) => <button className={page === key ? "nav-item active" : "nav-item"} key={key} onClick={() => navigate(key)}><span>{icon}</span>{label}</button>)}<div className="sidebar-footer"><span className="avatar">EC</span><div><strong>Marketing team</strong><small>Local workspace</small></div></div></aside>;
}

export function AppLayout({ page, navigate, children }: { page: string; navigate: (page: string) => void; children: ReactNode }) {
  return <div className="app-frame"><Header onMenu={() => document.querySelector(".sidebar")?.classList.toggle("open")} /><Sidebar page={page} navigate={navigate} open={false} /><main className="content">{children}</main></div>;
}
