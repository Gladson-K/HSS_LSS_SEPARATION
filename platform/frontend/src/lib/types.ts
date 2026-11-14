export interface GraphPoint {
  t: number;
  value: number;
}

export type Severity = "normal" | "info" | "warning" | "critical";

export interface DiagnosticGraph {
  id: string;
  title: string;
  description: string;
  x_label: string;
  y_label: string;
  unit: string;
  series: GraphPoint[];
  inference: string;
  confidence: number;
  severity: Severity;
  suggestions: string[];
  references: string[];
}

export interface SeparationResult {
  sample_rate: number;
  duration_s: number;
  hss_rms: number;
  lss_rms: number;
  hss_snr_db: number;
  lss_snr_db: number;
  heartbeat_bpm: number | null;
  breath_rate_bpm: number | null;
  dominant_diagnosis: string;
  diagnosis_confidence: number;
}

export interface AnalysisResponse {
  analysis_id: string;
  filename: string;
  created_at: string;
  separation: SeparationResult;
  graphs: DiagnosticGraph[];
  summary: string;
  lifestyle: string[];
  disclaimer: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  task: string;
  source: string;
  license: string;
  commercial_ok: boolean;
  parameters: string | null;
  modality: string;
  notes: string;
}

export interface AgentResponse {
  answer: string;
  provider: string;
  model: string;
  used_fallback: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  providers_available: string[];
  models_available: string[];
}
