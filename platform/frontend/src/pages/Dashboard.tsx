import { useState } from "react";
import { Activity } from "lucide-react";
import { UploadPanel } from "@/components/UploadPanel";
import { GraphNavigator } from "@/components/GraphNavigator";
import { SummaryPanel } from "@/components/SummaryPanel";
import { AgentChat } from "@/components/AgentChat";
import type { AnalysisResponse } from "@/lib/types";

export function Dashboard() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  return (
    <div className="space-y-5">
      {!result && (
        <UploadPanel onResult={setResult} />
      )}

      {result && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Activity className="h-4 w-4 text-brand-300" />
              <span className="font-medium">{result.filename}</span>
              <span className="text-slate-500">
                · {result.separation.duration_s.toFixed(1)}s @{" "}
                {result.separation.sample_rate} Hz
              </span>
            </div>
            <button
              onClick={() => setResult(null)}
              className="text-xs text-brand-300 underline-offset-4 hover:underline"
            >
              Analyze another recording
            </button>
          </div>

          <SummaryPanel result={result} />

          <div className="grid gap-5 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <GraphNavigator graphs={result.graphs} />
            </div>
            <AgentChat analysisId={result.analysis_id} />
          </div>
        </>
      )}
    </div>
  );
}
