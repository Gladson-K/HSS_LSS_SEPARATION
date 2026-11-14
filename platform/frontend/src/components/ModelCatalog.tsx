import { useEffect, useMemo, useState } from "react";
import { Boxes, Filter } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Badge } from "./ui/Badge";

const MODALITIES = ["all", "audio", "text", "image", "image/text"];

export function ModelCatalog() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modality, setModality] = useState("all");
  const [permissiveOnly, setPermissiveOnly] = useState(false);

  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  const filtered = useMemo(
    () =>
      models.filter(
        (m) =>
          (modality === "all" || m.modality === modality) &&
          (!permissiveOnly || m.commercial_ok),
      ),
    [models, modality, permissiveOnly],
  );

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-500/15 text-indigo-300">
            <Boxes className="h-4 w-4" />
          </div>
          <div>
            <CardTitle>Open-Source Model Catalog</CardTitle>
            <p className="text-xs text-slate-400">
              Permissive (Apache/MIT/BSD) models available for fine-tuning
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={permissiveOnly}
              onChange={(e) => setPermissiveOnly(e.target.checked)}
              className="h-3.5 w-3.5 accent-brand-500"
            />
            <Filter className="h-3 w-3" /> Commercial-OK only
          </label>
          <select
            value={modality}
            onChange={(e) => setModality(e.target.value)}
            className="h-8 rounded-lg border border-white/10 bg-slate-950/60 px-2 text-xs text-slate-200"
          >
            {MODALITIES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-white/10 bg-white/[0.03] p-3 transition hover:border-brand-400/40 hover:bg-white/[0.05]"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-sm font-semibold text-slate-100">{m.name}</div>
                  <div className="text-[11px] text-slate-400">{m.task}</div>
                </div>
                <Badge className="shrink-0">{m.modality}</Badge>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] ${
                    m.commercial_ok
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                      : "border-amber-500/30 bg-amber-500/10 text-amber-300"
                  }`}
                >
                  {m.license}
                </span>
                {m.parameters && <Badge>{m.parameters}</Badge>}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-400">{m.notes}</p>
              <p className="mt-1 truncate text-[10px] text-slate-500">{m.source}</p>
            </div>
          ))}
        </div>
        {!filtered.length && (
          <p className="py-6 text-center text-sm text-slate-400">No models match the filter.</p>
        )}
      </CardContent>
    </Card>
  );
}
