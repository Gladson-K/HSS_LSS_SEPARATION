import { useEffect, useState } from "react";
import { Activity, Boxes, Github, HeartPulse, Server, Wind } from "lucide-react";
import { Dashboard } from "./pages/Dashboard";
import { ModelCatalog } from "./components/ModelCatalog";
import { api } from "./lib/api";
import type { HealthResponse } from "./lib/types";

type Tab = "analyzer" | "models";

export default function App() {
  const [tab, setTab] = useState<Tab>("analyzer");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setOnline(true);
      })
      .catch(() => setOnline(false));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="relative grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-brand-500 to-emerald-500 shadow-glow">
              <HeartPulse className="h-5 w-5 text-white" />
              <Wind className="absolute -right-1 -top-1 h-4 w-4 text-sky-200" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold tracking-tight">CardioPulmo AI</h1>
                <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-slate-300">
                  v{health?.version ?? "0.1.0"}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Heart–Lung Sound Separation · NAS · Diagnostics
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-1">
            <button
              onClick={() => setTab("analyzer")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition ${
                tab === "analyzer"
                  ? "bg-brand-600 text-white"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Activity className="h-4 w-4" /> Analyzer
            </button>
            <button
              onClick={() => setTab("models")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition ${
                tab === "models"
                  ? "bg-brand-600 text-white"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Boxes className="h-4 w-4" /> Models
            </button>
          </nav>

          <div className="hidden items-center gap-2 text-xs text-slate-400 md:flex">
            <Server className="h-3.5 w-3.5" />
            <span
              className={`h-2 w-2 rounded-full ${
                online === null
                  ? "bg-slate-500"
                  : online
                    ? "bg-emerald-400"
                    : "bg-rose-400"
              }`}
            />
            {online === null ? "connecting" : online ? "API online" : "API offline"}
            <a
              href="https://github.com/Gladson-K/HeartLungSound_NAS_Separation"
              target="_blank"
              rel="noreferrer"
              className="ml-2 rounded-lg border border-white/10 p-1.5 hover:bg-white/10"
              aria-label="GitHub"
            >
              <Github className="h-4 w-4" />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {online === false && (
          <div className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            Backend API not reachable. Start it with{" "}
            <code className="rounded bg-black/30 px-1">uvicorn app.main:app --port 8000</code>{" "}
            (or <code className="rounded bg-black/30 px-1">docker compose up</code>).
          </div>
        )}
        {tab === "analyzer" ? <Dashboard /> : <ModelCatalog />}
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-8 text-center text-xs text-slate-500">
        Research-grade screening support only — not a medical device. ©{" "}
        {new Date().getFullYear()} CardioPulmo AI.
      </footer>
    </div>
  );
}
