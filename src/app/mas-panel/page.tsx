"use client";

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { useFilesRef, fileKey } from "@/hooks/useFilesRef";
import type { FileState } from "@/types";
import { Button } from "@/components/ui/button";
import { SAMPLE_DATASETS, sampleAsFile } from "@/lib/sample-data";
import { useAppStore } from "@/lib/store";
import { useSystemSettings } from "@/lib/hooks";
import { useRestoreSession } from "@/hooks/useRestoreSession";
import { AlertCircle, Plus, RotateCcw, Settings2, User, X } from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

import { useColumnSelection } from "@/hooks/useColumnSelection";
import {
  dispatchProcessRow,
  dispatchConsensusRow,
  dispatchAgentNetworkRow,
} from "@/lib/llm-dispatch";
import { useBatchProcessor } from "@/hooks/useBatchProcessor";
import { useProcessingStore } from "@/lib/processing-store";

import { SmartFileUpload, type FileStatus } from "@/components/tools/SmartFileUpload";
import { ColumnSelector } from "@/components/tools/ColumnSelector";
import { NoModelWarning } from "@/components/tools/NoModelWarning";
import { AIInstructionsSection } from "@/components/tools/AIInstructionsSection";
import { ExecutionPanel } from "@/components/tools/ExecutionPanel";
import { SingleRunButton } from "@/components/tools/SingleRunButton";
import { ResultsPanel } from "@/components/tools/ResultsPanel";
import { useAIInstructions, AI_INSTRUCTIONS_MARKER } from "@/hooks/useAIInstructions";
import { useSessionState, clearSessionKeys } from "@/hooks/useSessionState";
import { parseStructuredFile } from "@/lib/parse-file";
import Papa from "papaparse";
import { extractTextBrowser } from "@/lib/document-browser";

import { AgentConfigDialog } from "./AgentConfigDialog";
import {
  type Agent,
  avatarStyle,
  emptyAgent,
  makeAgentId,
} from "@/lib/agent-library";

import { WorkflowModeSelector } from "./WorkflowModeSelector";
import { WorkflowLayout } from "./WorkflowLayouts";
import { type StepStatus } from "./WorkflowStepCard";
import { OutputFormatSelector, type OutputFormat } from "./OutputFormatSelector";
import { DeliberationSettingsSection } from "./DeliberationSettingsSection";
import {
  type WorkflowMode,
  type WorkflowStep,
  type DeliberationSettings,
  DEFAULT_DELIBERATION_SETTINGS,
  STEP_MINIMUMS,
  composeStepSystemPrompt,
  emptyStep,
  migrateLegacyMode,
} from "./workflow-types";

type Row = Record<string, unknown>;

function providerLabel(id: string) {
  if (id === "lmstudio") return "LM Studio";
  if (id === "ollama") return "Ollama";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

function AgentCard({
  agent,
  onConfigure,
  onRemove,
  canRemove,
}: {
  agent: Agent;
  onConfigure: () => void;
  onRemove: () => void;
  canRemove: boolean;
}) {
  return (
    <div className="border rounded-lg p-4 relative flex gap-5">
      {canRemove && (
        <button
          onClick={onRemove}
          className="absolute top-3 right-3 h-7 w-7 rounded-md border bg-background flex items-center justify-center text-muted-foreground hover:text-destructive hover:border-destructive transition-colors shadow-sm"
          title="Remove agent"
        >
          <X className="h-[18px] w-[18px]" strokeWidth={2.5} />
        </button>
      )}

      <div
        className={`shrink-0 w-24 h-24 rounded-lg overflow-hidden bg-muted/40 flex items-center justify-center ${
          typeof agent.avatar === "number" ? "border" : "border border-dashed border-muted-foreground/40"
        }`}
      >
        {typeof agent.avatar === "number" ? (
          <div className="w-full h-full" style={avatarStyle(agent.avatar)} aria-hidden />
        ) : (
          <User className="h-10 w-10 text-muted-foreground/60" />
        )}
      </div>

      <div className="flex-1 min-w-0 flex flex-col gap-2">
        <div className="space-y-1 pr-6">
          <div className="text-sm font-semibold truncate">
            {agent.name || <span className="text-muted-foreground italic">Unnamed agent</span>}
          </div>
          <div className="text-xs text-muted-foreground font-mono truncate">
            {providerLabel(agent.providerId)} / {agent.model || "—"}
          </div>
        </div>

        <div className="flex flex-wrap gap-1 text-[10px]">
          {agent.category && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{agent.category}</span>}
          {agent.personalityStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{agent.personalityStyle}</span>}
          {agent.communicationStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{agent.communicationStyle}</span>}
          {agent.responseStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{agent.responseStyle}</span>}
        </div>

        <div className="mt-auto pt-1">
          <Button variant="outline" size="sm" className="gap-1.5 text-xs" onClick={onConfigure}>
            <Settings2 className="h-3.5 w-3.5" /> Configure
          </Button>
        </div>
      </div>
    </div>
  );
}

export default function AgentPanelPage() {
  const [fileStates, setFileStates] = useSessionState<FileState[]>("agentpanel_fileStates", []);
  const filesRef = useFilesRef();
  const [previewRows, setPreviewRows] = useState<Row[] | null>(null);
  const [agents, setAgents] = useSessionState<Agent[]>("agentpanel_agents", []);
  const [workflowMode, setWorkflowMode] = useSessionState<WorkflowMode | null>("agentpanel_mode", null);
  useEffect(() => {
    const migrated = migrateLegacyMode(workflowMode);
    if (migrated !== workflowMode) setWorkflowMode(migrated);
  }, [workflowMode, setWorkflowMode]);
  const [workflowSteps, setWorkflowSteps] = useSessionState<WorkflowStep[]>("agentpanel_steps", []);
  const [delibSettings, setDelibSettings] = useSessionState<DeliberationSettings>(
    "agentpanel_delib",
    DEFAULT_DELIBERATION_SETTINGS,
  );
  const [outputFormat, setOutputFormat] = useSessionState<OutputFormat>("agentpanel_outputFormat", "per-row");

  const [configuringId, setConfiguringId] = useState<string | null>(null);
  const [stepStatuses, setStepStatuses] = useState<Record<string, StepStatus>>({});

  const providers = useAppStore((state) => state.providers);
  const systemSettings = useSystemSettings();
  const [concurrency, setConcurrency] = useState(systemSettings.maxConcurrency);

  const availableProviders = useMemo(
    () => Object.values(providers).filter((p) => p.isLocal || !!p.apiKey),
    [providers]
  );
  // Providers that are toggled on in Settings AND ready to use (have a key or are local).
  // These are the only providers used to populate default agent cards.
  const enabledProviders = useMemo(
    () => Object.values(providers).filter((p) => p.isEnabled && (p.isLocal || !!p.apiKey)),
    [providers]
  );
  const firstId = enabledProviders[0]?.providerId ?? "openai";
  const firstModel = enabledProviders[0]?.defaultModel ?? "gpt-4o";

  const makeDefaultAgents = useCallback((): Agent[] => {
    const pool =
      enabledProviders.length > 0
        ? enabledProviders.map((p) => ({ providerId: p.providerId, model: p.defaultModel || "" }))
        : [{ providerId: firstId, model: firstModel }];
    // Always 4 default cards; cycle through enabled providers if fewer than 4 are configured.
    return Array.from({ length: 4 }, (_, i) => {
      const s = pool[i % pool.length];
      return emptyAgent({
        id: makeAgentId(),
        name: `Agent ${i + 1}`,
        providerId: s.providerId,
        model: s.model,
      });
    });
  }, [enabledProviders, firstId, firstModel]);

  const [hydrationDone, setHydrationDone] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setHydrationDone(true), 0);
    return () => clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (hydrationDone && agents.length === 0) setAgents(makeDefaultAgents());
  }, [hydrationDone, agents.length, makeDefaultAgents, setAgents]);

  // Pad workflow steps to the mode's default minimum when the mode changes.
  // Reconcilier = 1 reconciler + 3 workers. Only adds — never removes.
  // Session restore sets `skipPaddingOnceRef` so the user's saved step count is
  // honored exactly (e.g. a 3-card Reconcilier run restores to 3 cards, not 4).
  const skipPaddingOnceRef = React.useRef(false);
  useEffect(() => {
    if (!workflowMode) return;
    if (skipPaddingOnceRef.current) {
      skipPaddingOnceRef.current = false;
      return;
    }
    const min = STEP_MINIMUMS[workflowMode];
    setWorkflowSteps((prev) => {
      if (prev.length >= min) return prev;
      return [
        ...prev,
        ...Array.from({ length: min - prev.length }, () => emptyStep()),
      ];
    });
  }, [workflowMode, setWorkflowSteps]);

  const updateAgent = (id: string, updated: Agent) => {
    setAgents((prev) => prev.map((a) => (a.id === id ? updated : a)));
  };
  const addAgent = () => {
    setAgents((prev) => [
      ...prev,
      emptyAgent({ id: makeAgentId(), name: `Agent ${prev.length + 1}`, providerId: firstId, model: firstModel }),
    ]);
  };
  const removeAgent = (id: string) => {
    setAgents((prev) => prev.filter((a) => a.id !== id));
    // Also clear any workflow step that references it
    setWorkflowSteps((prev) => prev.map((s) => (s.agentId === id ? { ...s, agentId: null } : s)));
  };

  // Per-step live status during batch processing. Keyed by step.id.
  const markStepStatus = (id: string, status: StepStatus) => {
    setStepStatuses((prev) => ({ ...prev, [id]: status }));
  };

  const updateStep = (id: string, updated: WorkflowStep) => {
    setWorkflowSteps((prev) => prev.map((s) => (s.id === id ? updated : s)));
  };
  const addStep = () => setWorkflowSteps((prev) => [...prev, emptyStep()]);
  const removeStep = (id: string) => setWorkflowSteps((prev) => prev.filter((s) => s.id !== id));

  // File objects can't survive a reload
  useEffect(() => {
    if (fileStates.length > 0 && !filesRef.current.has(fileKey(fileStates[0].file))) {
      setFileStates([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hasFile = fileStates.length > 0;
  const hasStructuredFile = hasFile && !!previewRows && previewRows.length > 0;
  const hasUnstructuredFile = hasFile && !hasStructuredFile;
  const previewColumns = useMemo(
    () => (previewRows && previewRows.length > 0 ? Object.keys(previewRows[0]) : []),
    [previewRows],
  );

  const { selectedCols, setSelectedCols, toggleCol, toggleAll } = useColumnSelection(
    "agentpanel_selectedCols",
    previewColumns,
    false,
  );

  const data: Row[] = useMemo(() => {
    if (hasStructuredFile && outputFormat === "per-row" && previewRows) return previewRows;
    if (hasStructuredFile && outputFormat === "document") return [{ document_name: fileStates[0].file.name }];
    if (hasUnstructuredFile) return [{ document_name: fileStates[0].file.name }];
    return [{ _no_file: "instruction-only" }];
  }, [hasStructuredFile, hasUnstructuredFile, outputFormat, previewRows, fileStates]);
  const dataName = hasFile ? fileStates[0].file.name : "instruction-only";

  const agentById = useMemo(
    () => Object.fromEntries(agents.map((a) => [a.id, a])),
    [agents],
  );

  // ── Auto-generated AI Instructions (summary of the workflow) ──
  // Stringify previewRows separately so changes to other workflow inputs don't
  // re-serialize a potentially-large CSV snapshot. Only embedded in document
  // mode, where the per-row save is just `{document_name}` and restore has
  // nothing else to rebuild the structured data from.
  const previewRowsJson = useMemo(
    () =>
      outputFormat === "document" && previewRows && previewRows.length > 0
        ? JSON.stringify(previewRows)
        : null,
    [outputFormat, previewRows],
  );

  const buildAutoInstructions = useCallback(() => {
    const lines: string[] = [];
    lines.push("You are orchestrating a multi-agent workflow.");
    lines.push("");

    if (workflowMode) {
      lines.push(`WORKFLOW MODE: ${workflowMode}`);
      lines.push("");
    }

    if (selectedCols.length > 0) {
      lines.push("SELECTED COLUMNS:");
      selectedCols.forEach((c) => lines.push(`- ${c}`));
      lines.push("");
    }

    if (workflowSteps.length > 0) {
      lines.push("STEPS:");
      workflowSteps.forEach((s, i) => {
        const a = s.agentId ? agentById[s.agentId] : null;
        const agentRef = a ? `${a.name || "(unnamed)"} — ${providerLabel(a.providerId)}/${a.model}` : "(no agent selected)";
        lines.push(`- Step ${i + 1}: ${agentRef}${s.taskDescription ? ` — ${s.taskDescription.slice(0, 80)}` : ""}`);
      });
      lines.push("");
    }

    lines.push("RULES:");
    lines.push("- Process each row independently");
    lines.push("- Return only the final result, no explanation");
    lines.push("- Do not include markdown or code fences");
    lines.push("");

    // Single-line JSON: regex parsing on restore relies on `.+$` matching one line.
    const workflowConfigOpen = JSON.stringify({
      mode: workflowMode,
      agents,
      steps: workflowSteps,
      outputFormat,
      selectedCols,
      delibSettings,
    });
    const workflowConfig = previewRowsJson
      ? workflowConfigOpen.slice(0, -1) + `,"previewRows":${previewRowsJson}}`
      : workflowConfigOpen;
    lines.push("WORKFLOW CONFIG: " + workflowConfig);
    lines.push("");

    lines.push(AI_INSTRUCTIONS_MARKER);
    return lines.join("\n");
  }, [workflowMode, selectedCols, workflowSteps, agentById, outputFormat, delibSettings, agents, previewRowsJson]);

  const [aiInstructions, setAiInstructions] = useAIInstructions(buildAutoInstructions);

  // Representative model (first agent) — used by batch processor's activeModel contract
  const representativeModel = useMemo(() => {
    const first = agents[0];
    if (!first) return null;
    const prov = providers[first.providerId];
    if (!prov) return null;
    return { ...prov, providerId: first.providerId, defaultModel: first.model };
  }, [agents, providers]);

  // Validate the run
  const validate = (): string | null => {
    if (!workflowMode) return "Pick a workflow mode";
    if (workflowSteps.length < 2) return "Add at least 2 workflow steps";
    if (hasStructuredFile && selectedCols.length === 0) return "Select at least one column";
    for (let i = 0; i < workflowSteps.length; i++) {
      const s = workflowSteps[i];
      if (!s.agentId) return `Step ${i + 1} has no agent selected`;
      const agent = agentById[s.agentId];
      if (!agent) return `Step ${i + 1} references a missing agent`;
      const prov = providers[agent.providerId];
      if (!prov) return `Invalid provider for Step ${i + 1}`;
      if (!prov.isLocal && !prov.apiKey) return `API key missing for ${providerLabel(agent.providerId)} (Step ${i + 1})`;
      if (!agent.model.trim()) return `Model name required for Step ${i + 1}`;
    }
    return null;
  };

  // Build user content sent to the LLM.
  //   per-row:   subset of the current row's selected columns as JSON
  //   document:  JSON array of every row with selected columns — entire file is one document
  //   unstructured: extracted text from the file
  //   no file:   the full AI Instructions text
  const buildUserContent = async (row: Row): Promise<string> => {
    if (hasStructuredFile && outputFormat === "per-row") {
      const subset: Row = {};
      selectedCols.forEach((col) => (subset[col] = row[col]));
      return JSON.stringify(subset);
    }
    if (hasStructuredFile && outputFormat === "document" && previewRows) {
      const payload = previewRows.map((r) => {
        const subset: Row = {};
        selectedCols.forEach((col) => (subset[col] = r[col]));
        return subset;
      });
      return JSON.stringify(payload);
    }
    if (hasUnstructuredFile) {
      const file = fileStates[0]?.file;
      if (!file) throw new Error("File not found");
      const { text } = await extractTextBrowser(file);
      return text;
    }
    return aiInstructions;
  };

  // ── The mode-branching processor ───────────────────────────────────────────
  const processRow = async (row: Row): Promise<Row> => {
    const userContent = await buildUserContent(row);
    const baseRow: Row = hasStructuredFile
      ? row
      : hasUnstructuredFile
      ? { document_name: fileStates[0]?.file.name ?? "" }
      : {};

    // Resolve (step, agent) pairs
    const resolved = workflowSteps.map((s) => {
      const agent = agentById[s.agentId!];
      return { step: s, agent, prov: providers[agent.providerId] };
    });

    if (workflowMode === "sequential") {
      let currentInput = userContent;
      const outputs: Row = {};
      let totalLatencyMs = 0;
      for (let i = 0; i < resolved.length; i++) {
        const { step, agent, prov } = resolved[i];
        markStepStatus(step.id, "running");
        const systemPrompt = `${aiInstructions}\n\n${composeStepSystemPrompt(agent, step)}`;
        try {
          const result = await dispatchProcessRow({
            provider: agent.providerId,
            model: agent.model,
            apiKey: prov?.apiKey ?? "",
            baseUrl: prov?.baseUrl,
            systemPrompt,
            userContent: currentInput,
            temperature: systemSettings.temperature,
            maxTokens: systemSettings.maxTokens ?? undefined,
          });
          outputs[`step_${i + 1}_output`] = result.output;
          outputs[`step_${i + 1}_latency_ms`] = String(result.latency);
          totalLatencyMs += result.latency;
          currentInput = result.output;
          markStepStatus(step.id, "done");
        } catch (err) {
          markStepStatus(step.id, "error");
          throw err;
        }
      }
      return { ...baseRow, ...outputs, status: "success", latency_ms: totalLatencyMs };
    }

    if (workflowMode === "reconcilier") {
      const [reconciler, ...workers] = resolved;
      // All workers kick off in parallel inside the dispatch — mark them together.
      workers.forEach(({ step }) => markStepStatus(step.id, "running"));
      markStepStatus(reconciler.step.id, "pending");
      // Strip the auto-generated workflow metadata (STEPS list, WORKFLOW CONFIG
      // JSON, etc.) before sending to workers. Each worker must NOT see the
      // names/personas of its peers — the model will hallucinate peer outputs.
      // Only the user-editable extras (text after AI_INSTRUCTIONS_MARKER) reach
      // the LLM; per-step task lives in the worker's persona via composeStepSystemPrompt.
      const markerIdx = aiInstructions.indexOf(AI_INSTRUCTIONS_MARKER);
      const userExtras = markerIdx === -1
        ? ""
        : aiInstructions.slice(markerIdx + AI_INSTRUCTIONS_MARKER.length).trim();
      // Per-worker task list — RECONCILER-ONLY context. Goes into reconcilerPrompt
      // (which workers never see); the route appends only workerPrompt to each
      // worker's system prompt, so this stays out of worker prompts.
      const perWorkerInstructions = workers
        .map(({ step }, i) => {
          const task = step.taskDescription.trim();
          return `worker_${i + 1} instruction: ${task || "(no specific instruction)"}`;
        })
        .join("\n");
      const reconcilerOnlyContext = `Per-worker tasks (each worker received only its own; this is your reference for judging compliance):\n${perWorkerInstructions}`;
      const reconcilerPrompt = userExtras
        ? `${reconcilerOnlyContext}\n\n${userExtras}`
        : reconcilerOnlyContext;
      let result;
      try {
        result = await dispatchConsensusRow({
        workers: workers.map(({ step, agent, prov }) => ({
          provider: agent.providerId,
          model: agent.model,
          apiKey: prov?.apiKey ?? "",
          baseUrl: prov?.baseUrl,
          // Full per-step prompt (agent base + step persona + step knowledge +
          // step task) lives in `persona`. Each worker only sees its own.
          persona: composeStepSystemPrompt(agent, step),
        })),
        reconciler: {
          provider: reconciler.agent.providerId,
          model: reconciler.agent.model,
          apiKey: reconciler.prov?.apiKey ?? "",
          baseUrl: reconciler.prov?.baseUrl,
          persona: composeStepSystemPrompt(reconciler.agent, reconciler.step),
        },
        // workerPrompt is shared by every worker AND the reconciler — keep it
        // to user extras only so workers don't see peer task descriptions.
        workerPrompt: userExtras,
        // reconcilerPrompt is reconciler-only; safe place for the per-worker
        // task list.
        reconcilerPrompt,
        userContent,
        temperature: systemSettings.temperature,
        maxTokens: systemSettings.maxTokens ?? undefined,
      });
      } catch (err) {
        workers.forEach(({ step }) => markStepStatus(step.id, "error"));
        markStepStatus(reconciler.step.id, "error");
        throw err;
      }
      workers.forEach(({ step }) => markStepStatus(step.id, "done"));
      markStepStatus(reconciler.step.id, "done");
      const outputs: Row = {};
      const latencyCols: Row = {};
      (result.workerResults ?? []).forEach((w, i) => {
        outputs[`worker_${i + 1}_output`] = w.output;
        if (typeof w.latency === "number" && Number.isFinite(w.latency)) {
          latencyCols[`worker_${i + 1}_latency_ms`] = Math.round(w.latency * 1000);
        }
      });
      outputs["reconciler_output"] = result.reconcilerOutput;
      if (typeof result.reconcilerLatency === "number" && Number.isFinite(result.reconcilerLatency)) {
        latencyCols["reconciler_latency_ms"] = Math.round(result.reconcilerLatency * 1000);
      }
      // Wall-clock per row: workers run in parallel, then reconciler runs sequentially.
      // Latencies from ConsensusResult are in seconds; convert to ms.
      const workerLatencies = (result.workerResults ?? []).map((w) => w.latency);
      const maxWorker = workerLatencies.length > 0 ? Math.max(...workerLatencies) : 0;
      const latencyMs = Math.round((maxWorker + (result.reconcilerLatency ?? 0)) * 1000);
      return { ...baseRow, ...outputs, ...latencyCols, status: "success", latency_ms: latencyMs };
    }

    // deliberation
    resolved.forEach(({ step }) => markStepStatus(step.id, "running"));
    let result;
    try {
      result = await dispatchAgentNetworkRow({
      agents: resolved.map(({ step, agent, prov }, i) => ({
        label: agent.name || `Agent_${i + 1}`,
        role: composeStepSystemPrompt(agent, step),
        provider: agent.providerId,
        model: agent.model,
        apiKey: prov?.apiKey ?? "",
        baseUrl: prov?.baseUrl,
      })),
      userContent,
      maxRounds: delibSettings.maxRounds,
      // communicationStyle intentionally omitted — each agent carries its own via role
      convergenceMode: delibSettings.convergenceMode,
      convergenceThreshold: delibSettings.convergenceThreshold,
      temperature: systemSettings.temperature,
      maxTokens: systemSettings.maxTokens ?? undefined,
    });
    } catch (err) {
      resolved.forEach(({ step }) => markStepStatus(step.id, "error"));
      throw err;
    }
    resolved.forEach(({ step }) => markStepStatus(step.id, "done"));
    const outputs: Row = {
      rounds_taken: String(result.roundsTaken),
      converged: String(result.converged),
    };
    let totalLatencyMs = 0;
    (result.roundOutputs ?? []).forEach((r) => {
      const max = r.outputs.length > 0 ? Math.max(...r.outputs.map((o) => o.latency)) : 0;
      totalLatencyMs += max * 1000;
      r.outputs.forEach((o) => {
        outputs[`round_${r.round}_${o.label}_output`] = o.output;
      });
    });
    return { ...baseRow, ...outputs, status: "success", latency_ms: Math.round(totalLatencyMs) };
  };

  const batch = useBatchProcessor({
    toolId: "/mas-panel",
    runType: "agent-panel",
    activeModel: representativeModel,
    systemSettings,
    data,
    dataName,
    systemPrompt: aiInstructions,
    concurrency,
    validate,
    runParams: {
      provider: agents.map((a) => a.providerId).join(","),
      model: representativeModel?.defaultModel ?? "unknown",
      temperature: systemSettings.temperature,
    },
    processRow,
    buildResultEntry: (r: Row, i: number) => ({
      rowIndex: i,
      input: r as Record<string, unknown>,
      // Output intentionally blank — per-step / reconciler / final_consensus already
      // live in `input`, so repeating one of them here would create a duplicate
      // "output" column on the history detail page.
      output: "",
      status: (r.status as string) ?? (r.error_msg ? "error" : "success"),
      latency: r.latency_ms as number | undefined,
      errorMessage: r.error_msg as string | undefined,
    }),
  });

  // Column reorder: originals → step/worker/round outputs → final_output → step latencies → status → latency_ms → error_msg
  const displayResults = useMemo(() => {
    if (batch.results.length === 0) return batch.results;
    const allKeys = Object.keys(batch.results[0]);
    const meta = new Set(["status", "latency_ms", "error_msg"]);
    const FINAL_KEYS = new Set(["final_output", "final_consensus", "reconciler_output", "rounds_taken", "converged"]);
    const originalKeys = allKeys.filter((k) => !k.endsWith("_output") && !k.endsWith("_latency_ms") && !meta.has(k) && !FINAL_KEYS.has(k));
    const outputKeys = allKeys.filter((k) => k.endsWith("_output") && !FINAL_KEYS.has(k));
    const finalKeys = allKeys.filter((k) => FINAL_KEYS.has(k));
    const latencyKeys = allKeys.filter((k) => k.endsWith("_latency_ms"));
    const trailing = ["status", "latency_ms", "error_msg"].filter((k) => allKeys.includes(k));
    const ordered = [...originalKeys, ...outputKeys, ...finalKeys, ...latencyKeys, ...trailing];
    return batch.results.map((row) => {
      const out: Row = {};
      ordered.forEach((k) => { if (k in row) out[k] = row[k]; });
      return out;
    });
  }, [batch.results]);

  // Reset per-step statuses at the start of each run so stale colors don't linger.
  const wasProcessingRef = React.useRef(false);
  useEffect(() => {
    if (batch.isProcessing && !wasProcessingRef.current) {
      setStepStatuses({});
    }
    wasProcessingRef.current = batch.isProcessing;
  }, [batch.isProcessing]);

  // Session restore
  const restored = useRestoreSession("agent-panel");
  useEffect(() => {
    if (!restored) return;
    queueMicrotask(() => {
      // Parse WORKFLOW CONFIG first — we need `cfg.previewRows` as a fallback
      // for the file-rebuild step below when the saved per-row data is marker-
      // only (document-mode runs collapse every row into `{document_name}`).
      const fullPrompt = restored.systemPrompt ?? "";
      const configMatch = fullPrompt.match(/^WORKFLOW CONFIG: (.+)$/m);
      let cfg: {
        mode?: WorkflowMode | null;
        agents?: Agent[];
        steps?: WorkflowStep[];
        outputFormat?: OutputFormat;
        selectedCols?: string[];
        delibSettings?: DeliberationSettings;
        previewRows?: Row[];
      } = {};
      if (configMatch) {
        try {
          cfg = JSON.parse(configMatch[1]);
        } catch {
          // Malformed config — fall through with empty cfg; everything stays at defaults.
        }
      }

      // The padding effect would otherwise pad short saved configurations up to
      // the mode's default — signal a one-shot skip so the restored step count
      // is honored. Guarded on actual mode change: if the effect doesn't fire,
      // a stuck flag would skip the next user-initiated mode change.
      if (cfg.mode !== undefined) {
        const mode = migrateLegacyMode(cfg.mode);
        if (mode !== workflowMode) skipPaddingOnceRef.current = true;
        setWorkflowMode(mode);
      }
      if (Array.isArray(cfg.agents) && cfg.agents.length > 0) setAgents(cfg.agents);
      if (Array.isArray(cfg.steps)) setWorkflowSteps(cfg.steps);
      if (cfg.outputFormat) setOutputFormat(cfg.outputFormat);
      if (cfg.delibSettings) setDelibSettings(cfg.delibSettings);

      if (restored.dataName && restored.dataName !== "unnamed") {
        // Two recovery paths for previewRows:
        //   per-row mode → strip workflow output cols from each saved row to
        //                  recover the originals.
        //   document mode → per-row data is just `{document_name}`, so use the
        //                  snapshot embedded in WORKFLOW CONFIG (`cfg.previewRows`).
        // Falls back to a zero-byte placeholder for unstructured files /
        // instruction-only runs whose original bytes can't be reconstructed.
        const RESTORE_META = new Set([
          "status", "latency_ms", "error_msg",
          "final_output", "final_consensus", "reconciler_output",
          "rounds_taken", "converged",
        ]);
        const stripped: Row[] = restored.data.map((row) => {
          const clean: Row = {};
          for (const [k, v] of Object.entries(row)) {
            if (k.endsWith("_output") || k.endsWith("_latency_ms")) continue;
            if (RESTORE_META.has(k)) continue;
            clean[k] = v;
          }
          return clean;
        });
        const sampleKeys = Object.keys(stripped[0] ?? {});
        const isMarkerOnly =
          sampleKeys.length === 0 ||
          (sampleKeys.length === 1 && (sampleKeys[0] === "document_name" || sampleKeys[0] === "_no_file"));

        let recoveredRows: Row[] | null = null;
        if (!isMarkerOnly) {
          recoveredRows = stripped;
        } else if (Array.isArray(cfg.previewRows) && cfg.previewRows.length > 0) {
          recoveredRows = cfg.previewRows;
        }

        filesRef.current.clear();
        let restoredFile: File;
        if (recoveredRows) {
          const csv = Papa.unparse(recoveredRows);
          const restoredName = /\.(csv|xlsx|xls|json|ris)$/i.test(restored.dataName)
            ? restored.dataName.replace(/\.(xlsx|xls|json|ris)$/i, ".csv")
            : `${restored.dataName}.csv`;
          restoredFile = new File([csv], restoredName, { type: "text/csv" });
          setPreviewRows(recoveredRows);
        } else {
          restoredFile = new File([], restored.dataName);
          setPreviewRows(null);
        }
        filesRef.current.set(fileKey(restoredFile), restoredFile);
        setFileStates([{ file: restoredFile, status: "done" }]);
      }

      // useColumnSelection skips its auto-select-all when the existing selection
      // is a valid subset of the restored columns — so set this AFTER previewRows.
      if (Array.isArray(cfg.selectedCols) && cfg.selectedCols.every((c) => typeof c === "string")) {
        setSelectedCols(cfg.selectedCols);
      }

      const errors = restored.results.filter((r) => r.status === "error").length;
      useProcessingStore.getState().completeJob(
        "/mas-panel",
        restored.results,
        { success: restored.results.length - errors, errors, avgLatency: 0 },
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
  }, [batch, setFileStates, filesRef]);

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
  }, [batch, setFileStates, filesRef]);

  const currentFile = fileStates[0]?.file ?? null;
  const resultRow = batch.results[0];
  let fileStatus: FileStatus = "pending";
  if (batch.isProcessing) fileStatus = "processing";
  else if (resultRow?.status === "error") fileStatus = "error";
  else if (resultRow?.status === "success") fileStatus = "done";
  const fileError = resultRow?.error_msg as string | undefined;

  // Section numbering — everything shifts based on which optional sections render.
  const isDeliberation = workflowMode === "deliberation";
  const nCols = hasStructuredFile ? 2 : null;
  const nAgents = hasStructuredFile ? 3 : 2;
  const nMode = nAgents + 1;
  const nWorkflow = workflowMode ? nMode + 1 : null;
  // Output Format only shown when a structured file is uploaded AND a workflow mode is
  // selected (it sits after the Agent Workflow section). Unstructured files skip the
  // picker because they're document-mode by necessity.
  const nOutputFormat = hasStructuredFile && nWorkflow ? nWorkflow + 1 : null;
  const nDelib = isDeliberation && nWorkflow ? (nOutputFormat ?? nWorkflow) + 1 : null;
  const nAI = (nDelib ?? nOutputFormat ?? nWorkflow ?? nMode) + 1;
  const nExecute = nAI + 1;

  const handleStartOver = () => {
    clearSessionKeys("agentpanel_");
    filesRef.current.clear();
    setFileStates([]);
    setPreviewRows(null);
    setAgents(makeDefaultAgents());
    setWorkflowMode(null);
    setWorkflowSteps([]);
    setDelibSettings(DEFAULT_DELIBERATION_SETTINGS);
    setOutputFormat("per-row");
    setConcurrency(systemSettings.maxConcurrency);
    setAiInstructions("");
    batch.clearResults();
  };

  const configuringAgent = agents.find((a) => a.id === configuringId) ?? null;

  // Execution panel — shared config.
  //   per-row over a structured file → ExecutionPanel with row count
  //   document mode over a structured file OR any unstructured file → ExecutionPanel
  //     with unitLabel "file" so the progress UI reads "0/1 file".
  //   no file (instruction-only) → SingleRunButton
  const isDocumentMode = (hasStructuredFile && outputFormat === "document") || hasUnstructuredFile;
  const executePanel = hasFile ? (
    <ExecutionPanel
      isProcessing={batch.isProcessing}
      aborting={batch.aborting}
      runMode={batch.runMode}
      progress={batch.progress}
      etaStr={batch.etaStr}
      dataCount={data.length}
      disabled={validate() !== null}
      onRun={batch.run}
      onAbort={batch.abort}
      onResume={batch.resume}
      onCancel={batch.clearResults}
      failedCount={batch.failedCount}
      skippedCount={batch.skippedCount}
      {...(isDocumentMode
        ? { unitLabel: "file", hideTestButton: true, fullLabel: "Full run" }
        : { fullLabel: `Full Run (${data.length} rows)` })}
    />
  ) : (
    <SingleRunButton
      label="Run workflow"
      isProcessing={batch.isProcessing}
      aborting={batch.aborting}
      disabled={validate() !== null}
      onRun={() => batch.run("full")}
      onAbort={batch.abort}
    />
  );

  const aiInstructionsPanel = (sectionNumber: number) => (
    <AIInstructionsSection
      sectionNumber={sectionNumber}
      value={aiInstructions}
      onChange={setAiInstructions}
    >
      <NoModelWarning activeModel={availableProviders.length > 0 ? availableProviders[0] : null} />
      <div className="flex items-center gap-2 text-sm text-muted-foreground pt-2">
        <span>Concurrency:</span>
        <button className="px-2 py-1 border rounded hover:bg-muted transition-colors" onClick={() => setConcurrency(c => Math.max(1, c - 1))}>−</button>
        <span className="px-3 border-x min-w-[2rem] text-center">{concurrency}</span>
        <button className="px-2 py-1 border rounded hover:bg-muted transition-colors" onClick={() => setConcurrency(c => Math.min(10, c + 1))}>+</button>
        <span className="text-xs">(parallel API calls)</span>
      </div>
    </AIInstructionsSection>
  );

  return (
    <div className="space-y-0 pb-16">
      <div className="pb-6 flex items-start justify-between">
        <div className="space-y-1 max-w-3xl">
          <h1 className="text-4xl font-bold">MAS Panel</h1>
          <p className="text-muted-foreground text-sm">Unified multi-agent workflows — Reconcilier, Sequential, or Deliberation.</p>
        </div>
        <Button variant="destructive" className="gap-2 px-5" onClick={handleStartOver}>
          <RotateCcw className="h-3.5 w-3.5" /> Start Over
        </Button>
      </div>

      <div className={batch.isProcessing ? "pointer-events-none opacity-60" : ""}>

      {/* ── 1. Upload Data */}
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
              description="Choose which columns to send into the workflow for each row."
            />
          </div>
        </>
      )}

      <div className="border-t" />

      {/* ── N. Configure Agents */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">{nAgents}. Configure Agents</h2>
        <p className="text-sm text-muted-foreground">
          Define a pool of agents. Each agent can later be assigned to one or more workflow steps.
        </p>

        {availableProviders.length === 0 ? (
          <Link href="/settings">
            <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 cursor-pointer hover:opacity-90 text-sm text-amber-700">
              <AlertCircle className="h-4 w-4 shrink-0" />
              No providers with API keys configured — click here to go to Settings
            </div>
          </Link>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {agents.map((a) => (
                <AgentCard
                  key={a.id}
                  agent={a}
                  onConfigure={() => setConfiguringId(a.id)}
                  onRemove={() => removeAgent(a.id)}
                  canRemove={agents.length > 2}
                />
              ))}
            </div>
            <Button variant="outline" size="sm" className="text-xs" onClick={addAgent}>
              <Plus className="h-3.5 w-3.5 mr-1.5" /> Add Agent
            </Button>
          </>
        )}
      </div>

      <div className="border-t" />

      {/* ── N. Workflow Mode */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">{nMode}. Workflow Mode</h2>
        <WorkflowModeSelector value={workflowMode} onChange={setWorkflowMode} />
      </div>

      {workflowMode && (
        <>
          <div className="border-t" />

          {/* ── N. Agent Workflow */}
          <div className="space-y-4 py-8">
            <h2 className="text-2xl font-bold">{nWorkflow}. Agent Workflow</h2>
            <p className="text-sm text-muted-foreground">
              {workflowMode === "sequential" && "Steps run in order — output of one feeds the next."}
              {workflowMode === "reconcilier" && "The top card is the reconciler. Workers run in parallel; reconciler synthesizes their outputs."}
              {workflowMode === "deliberation" && "All agents deliberate over multiple rounds, each seeing the others' outputs."}
            </p>
            <WorkflowLayout
              mode={workflowMode}
              steps={workflowSteps}
              agents={agents}
              stepStatuses={stepStatuses}
              onUpdate={updateStep}
              onRemove={removeStep}
              onAdd={addStep}
            />
          </div>

          <div className="border-t" />

          {hasStructuredFile && nOutputFormat && (
            <>
              <div className="space-y-4 py-8">
                <h2 className="text-2xl font-bold">{nOutputFormat}. Output Format</h2>
                <OutputFormatSelector value={outputFormat} onChange={setOutputFormat} />
              </div>
              <div className="border-t" />
            </>
          )}

          {isDeliberation && nDelib && (
            <>
              <div className="space-y-4 py-8">
                <h2 className="text-2xl font-bold">{nDelib}. Deliberation Settings</h2>
                <DeliberationSettingsSection value={delibSettings} onChange={setDelibSettings} />
              </div>
              <div className="border-t" />
            </>
          )}
        </>
      )}

      {aiInstructionsPanel(nAI)}

      </div>

      <div className="border-t" />

      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">{nExecute}. Execute</h2>
        {executePanel}
      </div>

      <ResultsPanel
        results={displayResults}
        runId={batch.runId}
        title="Results"
        subtitle={`${displayResults.length} rows`}
      />

      {configuringAgent && (
        <AgentConfigDialog
          open={!!configuringId}
          onOpenChange={(open) => { if (!open) setConfiguringId(null); }}
          agent={configuringAgent}
          onSave={(updated) => updateAgent(configuringAgent.id, updated)}
          enabledProviders={enabledProviders}
        />
      )}
    </div>
  );
}
