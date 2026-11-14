import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Loader2, UploadCloud } from "lucide-react";
import { api } from "@/lib/api";
import type { AnalysisResponse } from "@/lib/types";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";

export function UploadPanel({
  onResult,
}: {
  onResult: (r: AnalysisResponse) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      const result = await api.analyze(file);
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="relative overflow-hidden">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
          drag ? "border-brand-400 bg-brand-500/10" : "border-white/15"
        }`}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="grid h-16 w-16 place-items-center rounded-2xl bg-brand-500/15 text-brand-300"
        >
          {busy ? (
            <Loader2 className="h-7 w-7 animate-spin" />
          ) : (
            <UploadCloud className="h-7 w-7" />
          )}
        </motion.div>
        <div>
          <p className="text-sm font-medium text-slate-200">
            Drop a stethoscope recording or browse
          </p>
          <p className="text-xs text-slate-400">
            WAV · MP3 · FLAC · OGG · M4A — up to 50 MB
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        <Button
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          size="md"
        >
          <Activity className="h-4 w-4" />
          {busy ? "Analyzing…" : "Select audio"}
        </Button>
        {error && (
          <p className="max-w-md rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
            {error}
          </p>
        )}
      </div>
    </Card>
  );
}
