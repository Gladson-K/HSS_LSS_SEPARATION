import type {
  AgentResponse,
  AnalysisResponse,
  HealthResponse,
  ModelInfo,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE || "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<HealthResponse>("/health"),
  models: () => req<ModelInfo[]>("/models"),
  analyze: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return req<AnalysisResponse>("/analyze", { method: "POST", body: form });
  },
  agent: (question: string, analysisId?: string) =>
    req<AgentResponse>("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, analysis_id: analysisId }),
    }),
};
