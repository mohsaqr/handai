"use client";

import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { useAppStore } from "@/lib/store";
import { useSystemSettings } from "@/lib/hooks";
import { useBatchProcessor } from "@/hooks/useBatchProcessor";
import { useRestoreSession } from "@/hooks/useRestoreSession";
import { useProcessingStore } from "@/lib/processing-store";
import { HelpCircle, Plus, X, RotateCcw, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { dispatchConsensusRow } from "@/lib/llm-dispatch";
import { SmartFileUpload, type FileStatus } from "@/components/tools/SmartFileUpload";
import { ColumnSelector } from "@/components/tools/ColumnSelector";
import { useColumnSelection } from "@/hooks/useColumnSelection";
import { AIInstructionsSection } from "@/components/tools/AIInstructionsSection";
import type { FileState } from "@/types";
import { extractTextBrowser } from "@/lib/document-browser";
import { SAMPLE_DATASETS, sampleAsFile } from "@/lib/sample-data";
import { parseStructuredFile } from "@/lib/parse-file";
import Papa from "papaparse";
import { useFilesRef, fileKey } from "@/hooks/useFilesRef";
import { useAIInstructions, AI_INSTRUCTIONS_MARKER } from "@/hooks/useAIInstructions";
import { useSessionState, clearSessionKeys } from "@/hooks/useSessionState";
import { ExecutionPanel } from "@/components/tools/ExecutionPanel";
import { SingleRunButton } from "@/components/tools/SingleRunButton";
import { ResultsPanel } from "@/components/tools/ResultsPanel";
import { OutputFormatSelector, type OutputFormat } from "@/app/mas-panel/OutputFormatSelector";

type Row = Record<string, unknown>;

const DEFAULT_WORKER_PROMPT = `You are an independent coder in an inter-rater reliability study. Code this text based on the instructions.

CODING RULES:
- Apply the codes the text genuinely speaks to — multi-coding is appropriate when multiple themes are present.
- Consider both explicit statements and implied meaning.
- Be consistent: apply the same standard to every text segment.
- Output ONLY the codes or values requested. No explanations, no commentary.
- Plain text only. No markdown, no headings, no code fences.`;

const DEFAULT_JUDGE_PROMPT = `You are a senior researcher judging between independent coders.

PROCEDURE:
1. Identify codes where all workers agree — accept these.
2. For disagreements, re-read the original text and evaluate against the codebook.
3. Favor inclusion when evidence is ambiguous but present.
4. Produce the final consolidated answer.

OUTPUT: Return ONLY the final answer. No explanations, no reasoning, no commentary.`;

const SAMPLE_JUDGE_PROMPTS: Record<string, string> = {
  "Majority vote": `Pick the answer that the majority of workers agree on. If there is a tie, pick the answer from the highest-ranked worker.\n\nRULES:\n- Output the majority answer directly\n- If tied, prefer Worker 1's answer\n- No explanations, no reasoning, no commentary`,
  "Best quality pick": `Evaluate each worker's output for accuracy, completeness, and clarity. Pick the single best response.\n\nRULES:\n- Output the best answer directly\n- No explanations, no reasoning, no commentary`,
  "Synthesize all": `Combine the best parts of all worker outputs into one comprehensive answer.\n\nRULES:\n- Merge insights from all workers into a single coherent response\n- Do not simply copy one worker — synthesize\n- No explanations, no reasoning, no commentary`,
  "Conservative merge": `Compare all worker responses and identify only the claims, codes, or findings that appear in at least two independent responses. Discard claims made by only one worker.\n\nRULES:\n- Extract discrete claims from each response\n- Only retain claims corroborated by 2+ workers\n- Synthesize retained claims into a coherent response\n- Prefer precision over recall\n- No explanations, no reasoning, no commentary`,
  "Devil's advocate": `For each worker response, identify the strongest possible counterargument or most critical flaw. Then select the response that best withstands this scrutiny.\n\nRULES:\n- Eliminate responses with factual errors or logical fallacies\n- Select the response with the least damaging weakness\n- If all responses have critical flaws, synthesize a corrected answer\n- Output the final answer directly\n- No explanations, no reasoning, no commentary`,
  "Iterative refinement": `Treat Worker 1's response as the initial draft. For each subsequent worker response, identify improvements — corrections, additions, or better phrasing — and apply them incrementally to build the best possible answer.\n\nRULES:\n- Start with Worker 1 as baseline\n- For each additional worker, only adopt changes that clearly improve accuracy or completeness\n- Do not degrade existing correct content\n- Output the final refined response\n- No explanations, no reasoning, no commentary`,
};

interface WorkerConfig {
  providerId: string;
  model: string;
  persona?: string;
}

interface KappaStats {
  kappa: number | null;
  kappaLabel: string;
}

function providerLabel(id: string) {
  if (id === "lmstudio") return "LM Studio";
  if (id === "ollama") return "Ollama";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

function WorkerCard({ label, cfg, setCfg, enabledProviders, onRemove }: {
  label: string;
  cfg: WorkerConfig;
  setCfg: (c: WorkerConfig) => void;
  enabledProviders: { providerId: string; defaultModel: string; isEnabled: boolean; apiKey?: string; baseUrl?: string; isLocal?: boolean }[];
  onRemove?: () => void;
}) {
  const [showPrompt, setShowPrompt] = useState(false);

  return (
    <div className="border rounded-lg p-4 space-y-3 relative">
      {onRemove && (
        <button onClick={onRemove} className="absolute top-3 right-3 text-muted-foreground hover:text-destructive" title={`Remove ${label}`}>
          <X className="h-4 w-4" />
        </button>
      )}

      {/* Label */}
      <div className="text-sm font-semibold text-muted-foreground">{label}</div>

      {/* Provider + Model */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Provider</Label>
          <Select value={cfg.providerId} onValueChange={(v) => setCfg({ ...cfg, providerId: v })}>
            <SelectTrigger className="text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {enabledProviders.map((p) => (
                <SelectItem key={p.providerId} value={p.providerId} className="text-xs">
                  {providerLabel(p.providerId)}
                </SelectItem>
              ))}
              {enabledProviders.length === 0 && (
                <SelectItem value={cfg.providerId} className="text-xs">No providers</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Model</Label>
          <Input
            value={cfg.model}
            onChange={(e) => setCfg({ ...cfg, model: e.target.value })}
            placeholder="e.g. gpt-4o"
            className="text-xs font-mono"
          />
        </div>
      </div>

      {/* Persona prompt (collapsible) */}
      <div>
        <button
          onClick={() => setShowPrompt(!showPrompt)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {showPrompt ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Persona prompt
        </button>
        {showPrompt && (
          <Textarea
            value={cfg.persona ?? ""}
            onChange={(e) => setCfg({ ...cfg, persona: e.target.value })}
            className="mt-2 min-h-[120px] text-xs font-mono resize-y"
            placeholder="Optional system prompt for this worker..."
          />
        )}
      </div>
    </div>
  );
}

function JudgeCard({ cfg, setCfg, enabledProviders }: {
  cfg: WorkerConfig;
  setCfg: (c: WorkerConfig) => void;
  enabledProviders: { providerId: string; defaultModel: string; isEnabled: boolean; apiKey?: string; baseUrl?: string; isLocal?: boolean }[];
}) {
  const [showPrompt, setShowPrompt] = useState(false);

  return (
    <div className="w-full max-w-2xl border-2 border-amber-400 bg-amber-50/30 dark:bg-amber-950/20 rounded-lg p-4 space-y-3">
      <div className="text-sm font-semibold text-amber-800 dark:text-amber-300">Judge</div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Provider</Label>
          <Select value={cfg.providerId} onValueChange={(v) => setCfg({ ...cfg, providerId: v })}>
            <SelectTrigger className="text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {enabledProviders.map((p) => (
                <SelectItem key={p.providerId} value={p.providerId} className="text-xs">
                  {providerLabel(p.providerId)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Model</Label>
          <Input
            value={cfg.model}
            onChange={(e) => setCfg({ ...cfg, model: e.target.value })}
            placeholder="e.g. gpt-4o"
            className="text-xs font-mono"
          />
        </div>
      </div>
      <div>
        <button
          onClick={() => setShowPrompt(!showPrompt)}
          className="flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400 hover:text-amber-900 dark:hover:text-amber-200"
        >
          {showPrompt ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Persona prompt
        </button>
        {showPrompt && (
          <Textarea
            value={cfg.persona ?? ""}
            onChange={(e) => setCfg({ ...cfg, persona: e.target.value })}
            className="mt-2 min-h-[120px] text-xs font-mono resize-y"
            placeholder="Optional system prompt for the judge..."
          />
        )}
      </div>
    </div>
  );
}

export default function ConsensusCoderPage() {
  const [fileStates, setFileStates] = useSessionState<FileState[]>("consensus_fileStates", []);
  const filesRef = useFilesRef();
  const [previewRows, setPreviewRows] = useState<Row[] | null>(null);

  // File objects can't survive a reload; drop any persisted entry whose File is gone.
  useEffect(() => {
    if (fileStates.length > 0 && !filesRef.current.has(fileKey(fileStates[0].file))) {
      setFileStates([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [workerPrompt, setWorkerPrompt] = useSessionState("consensus_workerPrompt", "");
  const [judgePrompt, setJudgePrompt] = useSessionState("consensus_judgePrompt", "");
  const [kappaStats, setKappaStats] = useSessionState<KappaStats | null>("consensus_kappaStats", null);
  // Accumulator for running cumulative kappa; reset on each batch when i === 0.
  // Lives in buildResultEntry which runs BEFORE dispatchSaveResults, so the
  // cumulative value gets persisted and survives a reload from history.
  const kappaAccRef = useRef<{ sum: number; count: number }>({ sum: 0, count: 0 });

  const [extraWorkers, setExtraWorkers] = useSessionState<WorkerConfig[]>("consensus_extraWorkers", []);
  const [includeJudgeReasoning, setIncludeJudgeReasoning] = useSessionState("consensus_includeJudgeReasoning", true);
  const [enableQualityScoring, setEnableQualityScoring] = useSessionState("consensus_enableQualityScoring", false);
  const [enableDisagreementAnalysis, setEnableDisagreementAnalysis] = useSessionState("consensus_enableDisagreementAnalysis", false);
  const [outputFormat, setOutputFormat] = useSessionState<OutputFormat>("consensus_outputFormat", "per-row");

  const providers = useAppStore((state) => state.providers);
  const systemSettings = useSystemSettings();
  // Providers that are toggled on in Settings AND ready to use (have a key or are local).
  // Drives both the card defaults and the provider dropdowns inside each card.
  const enabledProviders = useMemo(
    () => Object.values(providers).filter((p) => p.isEnabled && (p.isLocal || !!p.apiKey)),
    [providers],
  );
  const firstId = enabledProviders[0]?.providerId ?? "openai";
  const firstModel = enabledProviders[0]?.defaultModel ?? "gpt-4o";
  const secondId = enabledProviders[1]?.providerId ?? firstId;
  const secondModel = enabledProviders[1]?.defaultModel ?? firstModel;

  const [worker1, setWorker1] = useState<WorkerConfig>({ providerId: firstId, model: firstModel });
  const [worker2, setWorker2] = useState<WorkerConfig>({ providerId: secondId, model: secondModel });
  const [judge, setJudge] = useState<WorkerConfig>({ providerId: firstId, model: firstModel });

  // Once Zustand has hydrated the providers from localStorage, snap any worker/
  // judge whose saved provider isn't in the enabled-and-ready set to the first
  // enabled provider — so the default never points at an unconfigured provider.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (hydratedRef.current || enabledProviders.length === 0) return;
    hydratedRef.current = true;
    const enabledIds = new Set(enabledProviders.map((p) => p.providerId));
    const p1 = enabledProviders[0];
    const p2 = enabledProviders[1] ?? p1;
    setWorker1((prev) => enabledIds.has(prev.providerId) ? prev : { providerId: p1.providerId, model: p1.defaultModel });
    setWorker2((prev) => enabledIds.has(prev.providerId) ? prev : { providerId: p2.providerId, model: p2.defaultModel });
    setJudge((prev) => enabledIds.has(prev.providerId) ? prev : { providerId: p1.providerId, model: p1.defaultModel });
    setExtraWorkers((prev) =>
      prev.map((w) => enabledIds.has(w.providerId) ? w : { providerId: p1.providerId, model: p1.defaultModel })
    );
  }, [enabledProviders, setExtraWorkers]);

  const addWorker = () => setExtraWorkers((prev) => [...prev, { providerId: firstId, model: firstModel }]);
  const removeWorker = (idx: number) => setExtraWorkers((prev) => prev.filter((_, i) => i !== idx));
  const updateExtraWorker = (idx: number, cfg: WorkerConfig) => setExtraWorkers((prev) => prev.map((w, i) => (i === idx ? cfg : w)));

  const hasFile = fileStates.length > 0;
  const hasStructuredFile = hasFile && !!previewRows && previewRows.length > 0;
  const hasUnstructuredFile = hasFile && !hasStructuredFile;
  const isSingleRun = !hasFile;
  const previewColumns = previewRows && previewRows.length > 0 ? Object.keys(previewRows[0]) : [];
  const { selectedCols, setSelectedCols, toggleCol, toggleAll } = useColumnSelection("consensus_selectedCols", previewColumns);

  type RunMode = "structured" | "unstructured" | "single";
  const runMode: RunMode = hasStructuredFile ? "structured" : hasUnstructuredFile ? "unstructured" : "single";

  const data: Row[] = useMemo(() => {
    if (runMode === "structured" && outputFormat === "per-row" && previewRows) return previewRows;
    if (runMode === "structured" && outputFormat === "document" && fileStates[0]) return [{ document_name: fileStates[0].file.name }];
    return [{}];
  }, [runMode, outputFormat, previewRows, fileStates]);

  const dataName = hasFile ? fileStates[0].file.name : "single-run";

  const buildAutoInstructions = useCallback(() => {
    const lines: string[] = [];
    lines.push("Consensus coding: multiple workers analyze each row, a judge picks the best answer.");
    lines.push("");
    lines.push("RULES:");
    lines.push("- All outputs MUST be plain text or CSV. NEVER use markdown formatting: no **, no ## headings, no bullet points, no code blocks, no backticks.");
    lines.push("- Workers: apply instructions directly, return values only. Do NOT explain or justify.");
    lines.push("- Judge: pick the best answer and output it directly. May add one short sentence of reasoning if needed.");
    lines.push("- Keep all outputs short and precise. No extra text beyond what was requested.");
    lines.push("");

    if (workerPrompt.trim()) {
      lines.push("WORKER INSTRUCTIONS:");
      lines.push(workerPrompt.trim());
      lines.push("");
    }

    if (judgePrompt.trim()) {
      lines.push("JUDGE INSTRUCTIONS:");
      lines.push(judgePrompt.trim());
      lines.push("");
    }

    // Save Enhanced Judge Features state for restore
    const features: string[] = [];
    if (includeJudgeReasoning) features.push("judge_reasoning");
    if (enableQualityScoring) features.push("quality_scoring");
    if (enableDisagreementAnalysis) features.push("disagreement_analysis");
    if (features.length > 0) {
      lines.push("ENHANCED FEATURES: " + features.join(", "));
      lines.push("");
    }

    // Save worker/judge cards (provider+model+persona) so restore can rebuild them.
    // Single-line JSON: regex parsing on restore relies on `.+$` matching one line.
    const serialize = (w: WorkerConfig) => ({
      providerId: w.providerId,
      model: w.model,
      ...(w.persona ? { persona: w.persona } : {}),
    });
    const cardConfig = {
      workers: [worker1, worker2, ...extraWorkers].map(serialize),
      judge: serialize(judge),
    };
    lines.push("WORKERS CONFIG: " + JSON.stringify(cardConfig));
    lines.push("");

    // Save the user's column selection so restore can repopulate "Define Columns".
    // Stored on one line so the restore-side regex (.+$) matches cleanly.
    if (selectedCols.length > 0) {
      lines.push("SELECTED COLUMNS: " + JSON.stringify(selectedCols));
      lines.push("");
    }

    lines.push(AI_INSTRUCTIONS_MARKER);
    return lines.join("\n");
  }, [workerPrompt, judgePrompt, includeJudgeReasoning, enableQualityScoring, enableDisagreementAnalysis, worker1, worker2, judge, extraWorkers, selectedCols]);

  const [aiInstructions, setAiInstructions] = useAIInstructions(buildAutoInstructions);

  // Build a pseudo activeModel from the judge config for useBatchProcessor
  const judgeProvider = providers[judge.providerId];
  const activeModel = judgeProvider ? {
    ...judgeProvider,
    providerId: judge.providerId,
    defaultModel: judge.model,
  } : null;

  const batch = useBatchProcessor({
    toolId: "/model-comparison",
    runType: "model-comparison",
    activeModel,
    systemSettings,
    data,
    dataName,
    systemPrompt: aiInstructions,
    selectData: (_data: Row[], mode) => (mode === "test" ? _data.slice(0, 10) : _data),
    validate: () => {
      if (isSingleRun && !workerPrompt.trim()) {
        return "Upload files or write worker instructions";
      }
      const p1 = providers[worker1.providerId];
      const p2 = providers[worker2.providerId];
      const pR = providers[judge.providerId];
      if (!p1 || !p2 || !pR) return "Invalid provider selection";
      if ((!p1.isLocal && !p1.apiKey) || (!p2.isLocal && !p2.apiKey) || (!pR.isLocal && !pR.apiKey)) {
        return "API keys missing. Check Settings.";
      }
      for (let i = 0; i < extraWorkers.length; i++) {
        const ep = providers[extraWorkers[i].providerId];
        if (!ep) return `Invalid Worker ${i + 3} provider`;
        if (!ep.isLocal && !ep.apiKey) return `API key missing for Worker ${i + 3}. Check Settings.`;
      }
      return null;
    },
    runParams: {
      provider: judge.providerId,
      model: judge.model,
      temperature: systemSettings.temperature,
    },
    processRow: async (row: Row, idx: number) => {
      const p1 = providers[worker1.providerId];
      const p2 = providers[worker2.providerId];
      const pR = providers[judge.providerId];

      const workers = [
        { provider: worker1.providerId, model: worker1.model, apiKey: p1?.apiKey || "local", baseUrl: p1?.baseUrl, persona: worker1.persona },
        { provider: worker2.providerId, model: worker2.model, apiKey: p2?.apiKey || "local", baseUrl: p2?.baseUrl, persona: worker2.persona },
        ...extraWorkers.map((ew) => {
          const ep = providers[ew.providerId];
          return { provider: ew.providerId, model: ew.model, apiKey: ep?.apiKey || "local", baseUrl: ep?.baseUrl, persona: ew.persona };
        }),
      ];

      let userContent = "";
      let documentName = "";

      if (runMode === "single") {
        userContent = "";
      } else if (runMode === "unstructured") {
        const file = fileStates[0]?.file;
        if (!file) throw new Error("File not found");
        documentName = file.name;
        const { text } = await extractTextBrowser(file);
        if (!text.trim()) throw new Error("No text extracted from file");
        userContent = text;
      } else if (outputFormat === "document" && previewRows) {
        documentName = fileStates[0]?.file.name ?? "";
        const cols = selectedCols.length > 0 ? selectedCols : Object.keys(previewRows[0] ?? {});
        const payload = previewRows.map((r) => {
          const subset: Row = {};
          for (const c of cols) subset[c] = r[c];
          return subset;
        });
        userContent = JSON.stringify(payload);
      } else {
        const cols = selectedCols.length > 0 ? selectedCols : Object.keys(row);
        const subset: Row = {};
        for (const c of cols) subset[c] = row[c];
        userContent = JSON.stringify(subset);
      }

      const t0 = Date.now();
      const result = await dispatchConsensusRow({
        workers: workers.map((w) => ({ provider: w.provider, model: w.model, apiKey: w.apiKey || "", baseUrl: w.baseUrl, persona: w.persona })),
        reconciler: { provider: judge.providerId, model: judge.model, apiKey: pR?.apiKey || "", baseUrl: pR?.baseUrl, persona: judge.persona },
        workerPrompt: workerPrompt.trim() || DEFAULT_WORKER_PROMPT,
        reconcilerPrompt: judgePrompt.trim() || DEFAULT_JUDGE_PROMPT,
        userContent,
        enableQualityScoring,
        enableDisagreementAnalysis,
        includeReasoning: includeJudgeReasoning,
        temperature: systemSettings.temperature,
        maxTokens: systemSettings.maxTokens ?? undefined,
        rowIdx: idx,
      });
      const latencyMs = Date.now() - t0;

      const workerCols: Record<string, string> = {};
      const workerLatencyCols: Record<string, number> = {};
      result.workerResults?.forEach((wr: { output: string; latency: number }, i: number) => {
        workerCols[`worker_${i + 1}_output`] = wr?.output ?? "";
        if (typeof wr?.latency === "number" && Number.isFinite(wr.latency)) {
          workerLatencyCols[`worker_${i + 1}_latency_ms`] = Math.round(wr.latency * 1000);
        }
      });
      const judgeLatencyMs =
        typeof result.reconcilerLatency === "number" && Number.isFinite(result.reconcilerLatency)
          ? Math.round(result.reconcilerLatency * 1000)
          : null;

      const qualityCols: Record<string, string> = {};
      if (enableQualityScoring) {
        const scores = (result.qualityScores ?? []) as (number | null | undefined)[];
        for (let i = 0; i < workers.length; i++) {
          const s = scores[i];
          qualityCols[`quality_score_w${i + 1}`] =
            typeof s === "number" && Number.isFinite(s) ? String(s) : "";
        }
      }

      let baseRow: Row;
      if (runMode === "single") {
        baseRow = { prompt: workerPrompt.trim() };
      } else if (runMode === "unstructured") {
        baseRow = { document_name: documentName };
      } else {
        baseRow = row;
      }

      return {
        ...baseRow,
        ...workerCols,
        judge_output: result.reconcilerOutput,
        ...(includeJudgeReasoning ? { judge_reasoning: result.reconcilerReasoning ?? (result.consensusType === "Unanimous" ? "Same workers' outputs" : "") } : {}),
        consensus: result.consensusType,
        _row_kappa: result.kappa,
        kappa: "—",
        ...qualityCols,
        ...(enableDisagreementAnalysis ? { disagreement_reason: result.consensusType === "Unanimous" ? "No disagreement" : (result.disagreementReason ?? "") } : {}),
        ...workerLatencyCols,
        ...(judgeLatencyMs !== null ? { judge_latency_ms: judgeLatencyMs } : {}),
        status: "success",
        latency_ms: latencyMs,
      };
    },
    buildResultEntry: (r: Row, i: number) => {
      // Reset accumulator at the start of each batch (mergedResults.map calls
      // this in order with ascending i).
      if (i === 0) kappaAccRef.current = { sum: 0, count: 0 };
      if (r.status !== "error" && r.status !== "skipped") {
        const rk = r._row_kappa as number | null | undefined;
        if (rk !== null && rk !== undefined && !isNaN(rk)) {
          kappaAccRef.current.sum += rk;
          kappaAccRef.current.count++;
        }
      }
      // Mutate r so both the saved entry (via `input: r`) and the in-memory
      // display row carry the cumulative value.
      r.kappa = kappaAccRef.current.count > 0
        ? (kappaAccRef.current.sum / kappaAccRef.current.count).toFixed(3)
        : "—";
      return {
        rowIndex: i,
        input: r as Record<string, unknown>,
        // Output intentionally blank — judge_output already lives in `input`, so
        // repeating it here would create a duplicate "output" column in history.
        output: "",
        status: (r.consensus === "Error" ? "error" : "success") as string,
        latency: r.latency_ms as number | undefined,
        errorMessage: r.error_msg as string | undefined,
      };
    },
    onComplete: (results: Row[]) => {
      // Row-level `kappa` is already written in buildResultEntry. Here we only
      // compute the final cumulative value for the summary card.
      let sum = 0;
      let count = 0;
      for (const row of results) {
        if (row.status === "error" || row.status === "skipped") continue;
        const rk = row._row_kappa as number | null;
        if (rk !== null && rk !== undefined && !isNaN(rk)) {
          sum += rk;
          count++;
        }
      }

      if (count > 0) {
        const finalKappa = sum / count;
        let label = "Very Low";
        if (finalKappa >= 0.8) label = "Very High";
        else if (finalKappa >= 0.6) label = "High";
        else if (finalKappa >= 0.4) label = "Moderate";
        else if (finalKappa >= 0.2) label = "Low";
        setKappaStats({ kappa: finalKappa, kappaLabel: label });
      }
    },
  });

  // ── Session restore from history ───────────────────────────────────────────
  const restored = useRestoreSession(["model-comparison", "consensus-coder"]);
  React.useEffect(() => {
    if (!restored) return;
    queueMicrotask(() => {
      const fullPrompt = restored.systemPrompt ?? "";

      // Stop each section at the next known section header (or end of string),
      // not at the first blank line — user prompts may legitimately contain
      // blank lines (e.g. a "RULES:" block with a preceding blank line).
      const NEXT_SECTION = String.raw`(?=\n+(?:WORKER INSTRUCTIONS:|JUDGE INSTRUCTIONS:|RECONCILER INSTRUCTIONS:|ENHANCED FEATURES:|WORKERS CONFIG:|SELECTED COLUMNS:|Extra Instructions \(Optional\) :)|\s*$)`;
      const workerMatch = fullPrompt.match(new RegExp(String.raw`WORKER INSTRUCTIONS:\n([\s\S]*?)${NEXT_SECTION}`));
      setWorkerPrompt(workerMatch ? workerMatch[1].trim() : "");

      const judgeMatch = fullPrompt.match(new RegExp(String.raw`(?:JUDGE|RECONCILER) INSTRUCTIONS:\n([\s\S]*?)${NEXT_SECTION}`));
      if (judgeMatch) setJudgePrompt(judgeMatch[1].trim());

      const featuresMatch = fullPrompt.match(/ENHANCED FEATURES: (.+)/);
      if (featuresMatch) {
        const feats = featuresMatch[1].split(",").map((f) => f.trim());
        setIncludeJudgeReasoning(feats.includes("reconciler_reasoning") || feats.includes("judge_reasoning"));
        setEnableQualityScoring(feats.includes("quality_scoring"));
        setEnableDisagreementAnalysis(feats.includes("disagreement_analysis"));
      } else {
        setIncludeJudgeReasoning(true);
        setEnableQualityScoring(false);
        setEnableDisagreementAnalysis(false);
      }

      // Restore worker/judge cards. JSON is on one line (see buildAutoInstructions).
      // If the saved provider is no longer enabled, snap to the first enabled one
      // so the form never points at an unconfigured provider.
      const configMatch = fullPrompt.match(/^WORKERS CONFIG: (.+)$/m);
      if (configMatch) {
        try {
          const cfg = JSON.parse(configMatch[1]) as { workers?: WorkerConfig[]; judge?: WorkerConfig };
          const enabledIds = new Set(enabledProviders.map((p) => p.providerId));
          // If enabledProviders hasn't hydrated yet, keep raw values — the
          // hydration effect (above) will snap any disabled-provider workers
          // once providers load.
          const snap = (w: WorkerConfig): WorkerConfig =>
            enabledIds.size === 0 || enabledIds.has(w.providerId)
              ? w
              : { providerId: firstId, model: firstModel, ...(w.persona ? { persona: w.persona } : {}) };
          const ws = cfg.workers ?? [];
          if (ws[0]) setWorker1(snap(ws[0]));
          if (ws[1]) setWorker2(snap(ws[1]));
          setExtraWorkers(ws.slice(2).map(snap));
          if (cfg.judge) setJudge(snap(cfg.judge));
        } catch {
          // Malformed config — leave defaults in place.
        }
      }

      // Restore the user's column selection. Must run AFTER setPreviewRows
      // updates previewColumns, otherwise useColumnSelection's auto-select-all
      // logic overwrites the saved subset. Both state updates are queued in
      // this microtask so React batches them into one render — by the time
      // useColumnSelection re-evaluates, selectedCols is the restored subset
      // and its `selectedCols.every(c => allColumns.includes(c))` guard skips
      // the auto-select.
      const colsMatch = fullPrompt.match(/^SELECTED COLUMNS: (.+)$/m);
      if (colsMatch) {
        try {
          const cols = JSON.parse(colsMatch[1]);
          if (Array.isArray(cols) && cols.every((c) => typeof c === "string")) {
            setSelectedCols(cols);
          }
        } catch {
          // Malformed selection — leave default (all selected) in place.
        }
      }

      const latencies = restored.results
        .map((r) => r.latency_ms as number | undefined)
        .filter((l): l is number => l !== undefined && l > 0);
      const avgLatency = latencies.length > 0
        ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
        : 0;

      // Recompute cumulative Cohen's kappa from restored per-row kappas.
      // buildResultEntry spread the whole row into inputJson, so _row_kappa
      // survives the round-trip through the DB.
      let kappaSum = 0;
      let kappaCount = 0;
      for (const r of restored.results) {
        if (r.status === "error" || r.status === "skipped") {
          r.kappa = "—";
          continue;
        }
        const rk = r._row_kappa as number | null | undefined;
        if (rk !== null && rk !== undefined && !isNaN(rk)) {
          kappaSum += rk;
          kappaCount++;
        }
        r.kappa = kappaCount > 0 ? (kappaSum / kappaCount).toFixed(3) : "—";
      }
      if (kappaCount > 0) {
        const finalKappa = kappaSum / kappaCount;
        let label = "Very Low";
        if (finalKappa >= 0.8) label = "Very High";
        else if (finalKappa >= 0.6) label = "High";
        else if (finalKappa >= 0.4) label = "Moderate";
        else if (finalKappa >= 0.2) label = "Low";
        setKappaStats({ kappa: finalKappa, kappaLabel: label });
      } else {
        setKappaStats(null);
      }

      // Restore the file slot. For structured uploads (CSV/XLSX/JSON), also
      // reconstruct previewRows from the saved per-row inputs by stripping
      // the run's worker / judge / latency / meta columns. Document mode and
      // unstructured uploads save markers like `{document_name}` only — for
      // those, we leave previewRows null (the original PDF/DOCX bytes can't
      // be restored from history, so re-upload is required to re-run).
      if (restored.dataName && restored.dataName !== "single-run") {
        const RESTORE_META = new Set([
          "status", "latency_ms", "error_msg",
          "kappa", "_row_kappa", "consensus",
          "judge_reasoning", "disagreement_reason",
        ]);
        const stripped: Row[] = restored.data.map((row) => {
          const clean: Row = {};
          for (const [k, v] of Object.entries(row)) {
            if (k.endsWith("_output") || k.endsWith("_latency_ms")) continue;
            if (RESTORE_META.has(k)) continue;
            if (k.startsWith("quality_score_")) continue;
            clean[k] = v;
          }
          return clean;
        });
        const sampleKeys = Object.keys(stripped[0] ?? {});
        const isMarkerOnly =
          sampleKeys.length === 0 ||
          (sampleKeys.length === 1 && sampleKeys[0] === "document_name");

        filesRef.current.clear();
        // Structured upload: rebuild a real CSV File from the stripped rows so
        // the placeholder badge doesn't show — the data IS restored, even if
        // the original .xlsx bytes aren't. Document/unstructured paths fall
        // back to a zero-byte placeholder (user must re-upload to re-run).
        let restoredFile: File;
        if (!isMarkerOnly) {
          const csv = Papa.unparse(stripped);
          const restoredName = /\.(csv|xlsx|xls|json|ris)$/i.test(restored.dataName)
            ? restored.dataName.replace(/\.(xlsx|xls|json|ris)$/i, ".csv")
            : `${restored.dataName}.csv`;
          restoredFile = new File([csv], restoredName, { type: "text/csv" });
          setPreviewRows(stripped);
        } else {
          restoredFile = new File([], restored.dataName);
        }
        filesRef.current.set(fileKey(restoredFile), restoredFile);
        setFileStates([{ file: restoredFile, status: "done" }]);
      }

      const errors = restored.results.filter((r) => r.status === "error").length;
      useProcessingStore.getState().completeJob(
        "/model-comparison",
        restored.results as Row[],
        { success: restored.results.length - errors, errors, avgLatency },
        restored.runId,
      );
      const n = restored.results.length;
      toast.success(`Restored session from "${restored.dataName}" (${n} row${n === 1 ? "" : "s"})`);
    });
  }, [restored]);

  const adoptFile = useCallback((file: File, rows: Row[] | null) => {
    filesRef.current.clear();
    filesRef.current.set(fileKey(file), file);
    setFileStates([{ file, status: "pending" }]);
    setPreviewRows(rows && rows.length > 0 ? rows : null);
    batch.clearResults();
    setKappaStats(null);
  }, [batch, setFileStates, setKappaStats]);

  const handleDrop = useCallback(async (accepted: File[]) => {
    const file = accepted[0];
    if (!file) return;
    const rows = await parseStructuredFile(file);
    adoptFile(file, rows as Row[] | null);
  }, [adoptFile]);

  const handleLoadSample = useCallback((key: string) => {
    const made = sampleAsFile(key);
    if (!made) return;
    adoptFile(made.file, made.rows as Row[]);
    toast.success(`Loaded sample: ${SAMPLE_DATASETS[key].name}`);
  }, [adoptFile]);

  const handleClearFile = useCallback(() => {
    filesRef.current.clear();
    setFileStates([]);
    setPreviewRows(null);
    batch.clearResults();
    setKappaStats(null);
  }, [batch, setFileStates, setKappaStats]);

  const currentFile = fileStates[0]?.file ?? null;
  const resultRow = batch.results[0];
  let fileStatus: FileStatus = "pending";
  if (batch.isProcessing) fileStatus = "processing";
  else if (resultRow?.status === "error") fileStatus = "error";
  else if (resultRow?.status === "success") fileStatus = "done";
  const fileError = resultRow?.error_msg as string | undefined;

  // Dynamic section numbering — "Define Columns" + "Output Format" only appear for structured files.
  const nCols = hasStructuredFile ? 2 : null;
  const nConfigure = hasStructuredFile ? 3 : 2;
  const nOutputFormat = hasStructuredFile ? 4 : null;
  const nPrompts = hasStructuredFile ? 5 : 3;
  const nInstructions = hasStructuredFile ? 6 : 4;
  const nExecute = hasStructuredFile ? 7 : 5;

  const handleStartOver = useCallback(() => {
    clearSessionKeys("consensus_");
    filesRef.current.clear();
    setFileStates([]);
    setPreviewRows(null);
    setWorkerPrompt("");
    setJudgePrompt("");
    setKappaStats(null);
    setExtraWorkers([]);
    setIncludeJudgeReasoning(true);
    setEnableQualityScoring(false);
    setEnableDisagreementAnalysis(false);
    setOutputFormat("per-row");
    setAiInstructions("");
    batch.clearResults();
  }, [batch, setFileStates, setWorkerPrompt, setJudgePrompt, setKappaStats, setExtraWorkers, setIncludeJudgeReasoning, setEnableQualityScoring, setEnableDisagreementAnalysis, setOutputFormat, setAiInstructions]);

  return (
    <div className="space-y-0 pb-16">

      {/* Header */}
      <div className="pb-6 flex items-start justify-between">
        <div className="space-y-1 max-w-3xl">
          <h1 className="text-4xl font-bold">Model Comparison</h1>
          <p className="text-muted-foreground text-sm">Multi-model consensus coding with inter-rater reliability (Cohen&apos;s Kappa)</p>
        </div>
        <Button variant="destructive" className="gap-2 px-5" onClick={handleStartOver}>
            <RotateCcw className="h-3.5 w-3.5" /> Start Over
          </Button>
      </div>

      <div className={batch.isProcessing ? "pointer-events-none opacity-60" : ""}>
      {/* ── 1. Upload Data ────────────────────────────────────────────────── */}
      <div className="space-y-4 pb-8">
        <h2 className="text-2xl font-bold">1. Upload Data</h2>
        <SmartFileUpload
          file={currentFile}
          status={fileStatus}
          errorMessage={fileError}
          previewRows={previewRows}
          onDrop={handleDrop}
          onClear={handleClearFile}
          onLoadSample={handleLoadSample}
        />
      </div>

      {hasStructuredFile && (
        <>
          <div className="border-t" />
          <div className="space-y-4 py-8">
            <h2 className="text-2xl font-bold">{nCols}. Define Columns</h2>
            <ColumnSelector
              allColumns={previewColumns}
              selectedCols={selectedCols}
              onToggleCol={toggleCol}
              onToggleAll={toggleAll}
              description="Choose which columns of structured files are sent to the workers. Unstructured files (PDF/DOCX/TXT) are unaffected."
              emptyMessage="Upload a CSV or Excel file to see available columns."
            />
          </div>
        </>
      )}

      <div className="border-t" />

      {/* ── 3. Configure Workers & Judge ─────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">{nConfigure}. Configure Workers &amp; Judge</h2>

        <p className="text-sm text-muted-foreground">
          Each worker codes independently, then the judge resolves disagreements.
        </p>
        {/* Judge — centered */}
        <div className="flex justify-center">
          <JudgeCard cfg={judge} setCfg={setJudge} enabledProviders={enabledProviders} />
        </div>

        {/* Workers — 2 per row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <WorkerCard label="Worker 1" cfg={worker1} setCfg={setWorker1} enabledProviders={enabledProviders} />
          <WorkerCard label="Worker 2" cfg={worker2} setCfg={setWorker2} enabledProviders={enabledProviders} />
          {extraWorkers.map((ew, idx) => (
            <WorkerCard key={idx} label={`Worker ${idx + 3}`} cfg={ew} setCfg={(cfg) => updateExtraWorker(idx, cfg)} enabledProviders={enabledProviders} onRemove={() => removeWorker(idx)} />
          ))}
        </div>
        <Button variant="outline" size="sm" className="text-xs" onClick={addWorker}>
          <Plus className="h-3.5 w-3.5 mr-1.5" /> Add Worker
        </Button>
      </div>

      {hasStructuredFile && nOutputFormat && (
        <>
          <div className="border-t" />
          <div className="space-y-4 py-8">
            <h2 className="text-2xl font-bold">{nOutputFormat}. Output Format</h2>
            <OutputFormatSelector value={outputFormat} onChange={setOutputFormat} />
          </div>
        </>
      )}

      <div className="border-t" />

      {/* ── 4. Prompts ────────────────────────────────────────────────────── */}
      <div className="space-y-5 py-8">
        <h2 className="text-2xl font-bold">{nPrompts}. Define Instructions</h2>

        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between h-7">
              <Label className="text-sm">Worker Instructions</Label>
            </div>
            <Textarea value={workerPrompt} onChange={(e) => setWorkerPrompt(e.target.value)} className="min-h-[200px] text-xs font-mono resize-y" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between h-7">
              <Label className="text-sm">Judge Instructions</Label>
              <Select value="" onValueChange={(key) => { if (SAMPLE_JUDGE_PROMPTS[key]) setJudgePrompt(SAMPLE_JUDGE_PROMPTS[key]); }}>
                <SelectTrigger className="w-[160px] h-7 text-xs">
                  <SelectValue placeholder="Load sample..." />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(SAMPLE_JUDGE_PROMPTS).map((key) => (
                    <SelectItem key={key} value={key} className="text-xs">{key}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Textarea value={judgePrompt} onChange={(e) => setJudgePrompt(e.target.value)} className="min-h-[200px] text-xs font-mono resize-y" />
          </div>
        </div>

        <div className="space-y-3">
          <div className="text-sm font-bold">Enhanced Judge Features</div>
          <div className="grid grid-cols-3 gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={includeJudgeReasoning} onChange={(e) => setIncludeJudgeReasoning(e.target.checked)} className="accent-primary w-4 h-4" />
              <span className="text-sm">Include Judge Reasoning</span>
              <span title="Adds a column with the judge's reasoning for its choice">
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={enableQualityScoring} onChange={(e) => setEnableQualityScoring(e.target.checked)} className="accent-primary w-4 h-4" />
              <span className="text-sm">Quality Scoring</span>
              <span title="Judge assigns a quality score to each worker output">
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={enableDisagreementAnalysis} onChange={(e) => setEnableDisagreementAnalysis(e.target.checked)} className="accent-primary w-4 h-4" />
              <span className="text-sm">Disagreement Analysis</span>
              <span title="Adds a column explaining why workers disagreed">
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
              </span>
            </label>
          </div>
        </div>
      </div>

      <div className="border-t" />

      {/* ── 5. AI Instructions ─────────────────────────────────────────────── */}
      <AIInstructionsSection
        sectionNumber={nInstructions}
        value={aiInstructions}
        onChange={setAiInstructions}
      />

      </div>

      <div className="border-t" />

      {/* ── 6. Execute ────────────────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">{nExecute}. Execute</h2>
        {hasStructuredFile ? (
          <ExecutionPanel
            isProcessing={batch.isProcessing}
            aborting={batch.aborting}
            runMode={batch.runMode}
            progress={batch.progress}
            etaStr={batch.etaStr}
            dataCount={data.length}
            disabled={false}
            onRun={batch.run}
            onAbort={batch.abort}
            onResume={batch.resume}
            onCancel={batch.clearResults}
            failedCount={batch.failedCount}
            skippedCount={batch.skippedCount}
            {...(outputFormat === "document"
              ? { unitLabel: "file", hideTestButton: true, fullLabel: "Full run" }
              : {})}
          />
        ) : hasUnstructuredFile ? (
          <ExecutionPanel
            isProcessing={batch.isProcessing}
            aborting={batch.aborting}
            runMode={batch.runMode}
            progress={batch.progress}
            etaStr={batch.etaStr}
            dataCount={fileStates.length}
            disabled={false}
            onRun={batch.run}
            onAbort={batch.abort}
            onResume={batch.resume}
            onCancel={batch.clearResults}
            failedCount={batch.failedCount}
            skippedCount={batch.skippedCount}
            unitLabel="file"
            hideTestButton
            fullLabel="Process All"
          />
        ) : (
          <SingleRunButton
            label="Run"
            isProcessing={batch.isProcessing}
            aborting={batch.aborting}
            disabled={isSingleRun && !workerPrompt.trim()}
            onRun={() => batch.run("full")}
            onAbort={batch.abort}
          />
        )}
      </div>

      {/* ── Results ────────────────────────────────────────────────────────── */}
      <ResultsPanel
        results={batch.results}
        runId={batch.runId}
        title="Results"
        subtitle={`${batch.results.length} rows coded`}
      >
        {kappaStats && (
          <div className="flex items-center gap-8 px-5 py-4 rounded-lg border border-purple-200 bg-purple-50/30 dark:bg-purple-950/20">
            <div>
              <div className="text-[11px] text-muted-foreground uppercase tracking-wider font-medium">Cohen&apos;s Kappa</div>
              <div className="text-3xl font-bold text-purple-600 mt-0.5">
                {kappaStats.kappa !== null ? kappaStats.kappa.toFixed(3) : "N/A"}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-muted-foreground uppercase tracking-wider font-medium">Agreement Level</div>
              <div className="text-base font-semibold mt-0.5">{kappaStats.kappaLabel}</div>
            </div>
            <div className="flex-1 text-xs text-muted-foreground">
              Kappa measures inter-rater agreement beyond chance. 0 = chance, 1 = perfect, &lt;0 = less than chance.
            </div>
            {batch.runMode !== "full" && batch.results.length > 0 && (
              <span className="text-xs font-medium text-purple-600 border border-purple-200 px-2 py-0.5 rounded bg-purple-50 shrink-0">
                {batch.runMode === "preview" ? "Preview" : "Test"} run · {batch.results.length} of {data.length} rows
              </span>
            )}
          </div>
        )}
      </ResultsPanel>
    </div>
  );
}
