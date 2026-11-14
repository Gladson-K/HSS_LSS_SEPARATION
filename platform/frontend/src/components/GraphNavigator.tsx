import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  Lightbulb,
  Stethoscope,
  TrendingUp,
} from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DiagnosticGraph } from "@/lib/types";
import { Button } from "./ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { SeverityBadge } from "./ui/Badge";

export function GraphNavigator({ graphs }: { graphs: DiagnosticGraph[] }) {
  const [i, setI] = useState(0);
  if (!graphs.length) return null;
  const g = graphs[i];
  const chartData = g.series.map((p) => ({ t: p.t, v: p.value }));

  function go(dir: -1 | 1) {
    setI((prev) => (prev + dir + graphs.length) % graphs.length);
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-brand-500/15 text-brand-300">
            <Stethoscope className="h-4 w-4" />
          </div>
          <div>
            <CardTitle>{g.title}</CardTitle>
            <p className="text-xs text-slate-400">{g.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <SeverityBadge severity={g.severity} />
          <span className="text-xs text-slate-400">
            {i + 1} / {graphs.length}
          </span>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={g.id}
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -24 }}
              transition={{ duration: 0.25 }}
              className="h-72 w-full rounded-xl border border-white/5 bg-slate-950/40 p-2"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <defs>
                    <linearGradient id="gline" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#3385fc" />
                      <stop offset="100%" stopColor="#10b981" />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="t"
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickFormatter={(v) => Number(v).toFixed(2)}
                    stroke="#334155"
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    stroke="#334155"
                    width={48}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(15,23,42,0.95)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    labelFormatter={(v) => `${g.x_label}: ${Number(v).toFixed(3)}`}
                    formatter={(v: number) => [v.toFixed(4), g.y_label]}
                  />
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="url(#gline)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>
          </AnimatePresence>

          <button
            aria-label="Previous graph"
            onClick={() => go(-1)}
            className="absolute left-2 top-1/2 grid h-11 w-11 -translate-y-1/2 place-items-center rounded-full border border-white/10 bg-slate-900/80 text-slate-200 backdrop-blur transition hover:bg-brand-600 hover:text-white"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
          <button
            aria-label="Next graph"
            onClick={() => go(1)}
            className="absolute right-2 top-1/2 grid h-11 w-11 -translate-y-1/2 place-items-center rounded-full border border-white/10 bg-slate-900/80 text-slate-200 backdrop-blur transition hover:bg-brand-600 hover:text-white"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-brand-300">
              <TrendingUp className="h-3.5 w-3.5" /> Inference
            </div>
            <p className="text-sm leading-relaxed text-slate-200">{g.inference}</p>
            <p className="mt-2 text-xs text-slate-400">
              Confidence {(g.confidence * 100).toFixed(0)}%
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-emerald-300">
              <Lightbulb className="h-3.5 w-3.5" /> Suggestions
            </div>
            <ul className="space-y-1.5">
              {g.suggestions.map((s, idx) => (
                <li key={idx} className="flex gap-2 text-sm text-slate-300">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                  {s}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 pt-1">
          <Button variant="secondary" size="sm" onClick={() => go(-1)}>
            <ChevronLeft className="h-4 w-4" /> Previous
          </Button>
          <div className="flex flex-wrap justify-center gap-1.5">
            {graphs.map((gg, idx) => (
              <button
                key={gg.id}
                onClick={() => setI(idx)}
                aria-label={`Go to ${gg.title}`}
                className={`h-2 w-2 rounded-full transition ${
                  idx === i ? "w-5 bg-brand-400" : "bg-white/25 hover:bg-white/50"
                }`}
              />
            ))}
          </div>
          <Button variant="secondary" size="sm" onClick={() => go(1)}>
            Next <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
