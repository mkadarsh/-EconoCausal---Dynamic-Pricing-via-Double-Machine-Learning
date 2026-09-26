import { useEffect, useState } from "react";
import { AppLayout } from "./components/Layout";
import { AuditPage, CausalPage, CustomerPage, Dashboard, OptimizationPage, UploadPage } from "./pages";

const pages: Record<string, React.ComponentType<{ navigate: (page: string) => void }>> = {
  dashboard: Dashboard,
  upload: UploadPage,
  causal: CausalPage,
  customers: CustomerPage,
  optimization: OptimizationPage,
  audit: AuditPage,
};

function App() {
  const [page, setPage] = useState(window.location.hash.slice(1) || "dashboard");
  useEffect(() => { const onHash = () => setPage(window.location.hash.slice(1) || "dashboard"); window.addEventListener("hashchange", onHash); return () => window.removeEventListener("hashchange", onHash); }, []);
  const navigate = (next: string) => { window.location.hash = next; setPage(next); };
  const Page = pages[page] ?? Dashboard;
  return <AppLayout page={page} navigate={navigate}><Page navigate={navigate} /></AppLayout>;
}

export default App;
