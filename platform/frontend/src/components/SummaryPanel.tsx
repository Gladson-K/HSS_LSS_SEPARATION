import { HeartPulse, ShieldCheck, Wind } from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Badge } from "./ui/Badge";

function Metric({
  icon,
  label,
  value,
  tone = "text-slate-100",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-400">
        {icon}
        {label}
      </div>
      <div className={`text-lg font-semibold ${tone}`}>{value}</div>
    </div>
  );
}

export function SummaryPanel({ result }: { result: AnalysisResponse }) {
  const s = result.separation;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Analysis Summary</CardTitle>
          <Badge>{new Date(result.created_at).toLocaleString()}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Metric
            icon={<HeartPulse className="h-3.5 w-3.5 text-rose-400" />}
            label="Heart rate"
            value={s.heartbeat_bpm ? `${s.heartbeat_bpm.toFixed(0)} BPM` : "—"}
          />
          <Metric
            icon={<Wind className="h-3.5 w-3.5 text-sky-400" />}
            label="Respiratory"
            value={s.breath_rate_bpm ? `${s.breath_rate_bpm.toFixed(0)} /min` : "—"}
          />
          <Metric
            icon={<ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />}
            label="Screen"
            value={s.dominant_diagnosis}
            tone="text-emerald-300"
          />
          <Metric
            icon={<span className="text-xs">dB</span>}
            label="Isolation SNR"
            value={`${s.hss_snr_db.toFixed(1)} / ${s.lss_snr_db.toFixed(1)}`}
          />
        </div>

        <p className="text-sm leading-relaxed text-slate-300">{result.summary}</p>

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-300">
            Lifestyle suggestions
          </div>
          <ul className="grid gap-2 sm:grid-cols-2">
            {result.lifestyle.map((l, i) => (
              <li
                key={i}
                className="flex gap-2 rounded-lg border border-white/5 bg-white/[0.02] p-2.5 text-sm text-slate-300"
              >
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                {l}
              </li>
            ))}
          </ul>
        </div>

        <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs leading-relaxed text-amber-200">
          {result.disclaimer}
        </p>
      </CardContent>
    </Card>
  );
}
