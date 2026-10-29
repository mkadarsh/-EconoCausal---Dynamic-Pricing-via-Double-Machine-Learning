import { ReactNode, useEffect, useState } from "react";
import { api } from "../services/api";

type NavItem = { key: string; label: string; icon: string; route: string };

export const navItems: NavItem[] = [
  { key: "dashboard", label: "Overview", icon: "⌂", route: "dashboard" },
  { key: "campaign", label: "Campaign Lab", icon: "◈", route: "upload" },
  { key: "causal", label: "Causal Explorer", icon: "◎", route: "causal" },
  { key: "customers", label: "Customer Intelligence", icon: "◌", route: "customers" },
  { key: "optimization", label: "Budget Optimizer", icon: "↗", route: "optimization" },
  { key: "experimentation", label: "Experimentation", icon: "◇", route: "causal" },
  { key: "audit", label: "Causal Audit", icon: "✓", route: "audit" },
  { key: "observatory", label: "Data Observatory", icon: "▦", route: "upload" },
];

const titles: Record<string, string> = {
  dashboard: "Overview",
  upload: "Campaign Lab",
  causal: "Causal Explorer",
  customers: "Customer Intelligence",
  optimization: "Budget Optimizer",
  audit: "Causal Audit",
};

function Brand() {
  return <div className="brand"><div className="brand-mark">EC</div><div><strong>ECONO<span>CAUSAL</span></strong><small>Causal intelligence platform</small></div></div>;
}

export function Header({ page, onMenu, light, onTheme, connected }: { page: string; onMenu: () => void; light: boolean; onTheme: () => void; connected: boolean }) {
  return <header className="topbar">
    <button className="mobile-menu" onClick={onMenu} aria-label="Open navigation">☰</button>
    <div className="mobile-brand"><Brand /></div>
    <div className="breadcrumbs"><span>Workspace</span><b>/</b><strong>{titles[page] ?? "Overview"}</strong></div>
    <div className="topbar-tools">
      <label className="global-search"><span>⌕</span><input aria-label="Search workspace" placeholder="Search workspace" /></label>
      <span className={`connection-status ${connected ? "" : "disconnected"}`}><i /> API {connected ? "connected" : "unavailable"}</span>
      <button className="icon-button" onClick={onTheme} aria-label={light ? "Enable dark mode" : "Enable light mode"}>{light ? "☾" : "☼"}</button>
      <button className="avatar" aria-label="Workspace settings">EC</button>
    </div>
  </header>;
}

export function Sidebar({ page, navigate, open, close, connected }: { page: string; navigate: (page: string) => void; open: boolean; close: () => void; connected: boolean }) {
  return <aside className={`sidebar ${open ? "open" : ""}`}>
    <div className="sidebar-brand"><Brand /><button className="drawer-close" onClick={close} aria-label="Close navigation">×</button></div>
    <div className="side-label">INTELLIGENCE WORKSPACE</div>
    <nav aria-label="Primary navigation">{navItems.map((item) => <button className={page === item.route || (item.route === "upload" && page === "upload") ? "nav-item active" : "nav-item"} key={item.key} onClick={() => { navigate(item.route); close(); }}><span className="nav-icon">{item.icon}</span><span>{item.label}</span>{item.key === "causal" && <em>LIVE</em>}</button>)}</nav>
    <div className="sidebar-footer">
      <div className={`sidebar-status ${connected ? "" : "disconnected"}`}><i /> Backend {connected ? "connected" : "unavailable"}</div>
      <div className="dataset-chip"><span>DATASET</span><strong>Current campaign</strong><small>Local workspace</small></div>
      <small className="version">EconoCausal v0.1.0</small>
    </div>
  </aside>;
}

export function AppLayout({ page, navigate, children }: { page: string; navigate: (page: string) => void; children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [light, setLight] = useState(false);
  const [connected, setConnected] = useState(false);
  useEffect(() => { api.getHealth().then(() => setConnected(true)).catch(() => setConnected(false)); }, []);
  return <div className={`app-frame ${light ? "light-theme" : ""}`}>
    <Header page={page} onMenu={() => setDrawerOpen(true)} light={light} onTheme={() => setLight((value) => !value)} connected={connected} />
    <Sidebar page={page} navigate={navigate} open={drawerOpen} close={() => setDrawerOpen(false)} connected={connected} />
    {drawerOpen && <button className="drawer-backdrop" onClick={() => setDrawerOpen(false)} aria-label="Close navigation" />}
    <main className="content">{children}</main>
  </div>;
}
