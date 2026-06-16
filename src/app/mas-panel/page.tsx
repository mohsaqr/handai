"use client";

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { useFilesRef, fileKey } from "@/hooks/useFilesRef";
import type { FileState } from "@/types";
import { Button } from "@/components/ui/button";
import { SAMPLE_DATASETS, sampleAsFile } from "@/lib/sample-data";
import { useAppStore } from "@/lib/store";
import { useSystemSettings } from "@/lib/hooks";
import { useRestoreSession } from "@/hooks/useRestoreSession";
import { AlertCircle, Copy, HelpCircle, Plus, RotateCcw, Settings2, User, X } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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
  normalizeAgent,
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
  MAX_AGENTS_PER_LINE,
  composeStepSystemPrompt,
  emptyStep,
  migrateLegacyMode,
  buildStepLabels,
  wouldCreateCycle,
  topoOrder,
  includedColumns,
  AGENT_ROLE_PRESETS,
} from "./workflow-types";

type Row = Record<string, unknown>;

function providerLabel(id: string) {
  if (id === "lmstudio") return "LM Studio";
  if (id === "ollama") return "Ollama";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

// Judge mode (reconcilier) — which pool agent should sit in the top "judge" card.
// Pick by role preference; fall back to the first agent if none match.
const JUDGE_ROLE_PRIORITY = ["Judge", "Manager", "Synthesizer", "Critic", "Worker"];
function pickJudgeIndex(agents: Agent[]): number {
  for (const role of JUDGE_ROLE_PRIORITY) {
    const idx = agents.findIndex((a) => a.role === role);
    if (idx !== -1) return idx;
  }
  return 0;
}

// Default workflow = one card per configured agent. Personalized lays them across
// lines of MAX_AGENTS_PER_LINE; Judge mode promotes the best-matching agent (see
// pickJudgeIndex) to the top card; Sequential and Deliberation keep the Configure
// Agents pool order. This is what makes the workflow mirror the pool — 2 agents →
// 2 cards, etc.
function deriveStepsFromPool(mode: WorkflowMode, agents: Agent[]): WorkflowStep[] {
  if (mode === "reconcilier") {
    // Always show the judge card by default, even with an empty pool (the top
    // card is steps[0]) — equivalent to having clicked "Add worker" once.
    if (agents.length === 0) return [emptyStep()];
    const ji = pickJudgeIndex(agents);
    const ordered = [agents[ji], ...agents.filter((_, i) => i !== ji)];
    return ordered.map((a) => emptyStep({ agentId: a.id }));
  }
  // Sequential, Deliberation: keep the pool order so step N == the Nth agent the
  // user configured. (Personalized adds line/slot for its grid placement.)
  return agents.map((a, i) =>
    mode === "personalized"
      ? emptyStep({
          agentId: a.id,
          line: Math.floor(i / MAX_AGENTS_PER_LINE),
          slot: i % MAX_AGENTS_PER_LINE,
        })
      : emptyStep({ agentId: a.id }),
  );
}

// Judge (reconcilier), Individual (personalized) and Deliberation cards are a pure
// live projection of the Configure Agents pool: you add/remove agents there, never
// on the canvas (no "add" button, no per-card agent picker). This keeps the step
// list in lockstep with the pool while PRESERVING each surviving card's edits
// (connections via `inputs`, cut Judge spokes via `judgeExcluded`, per-card
// columns, persona): keep every step whose agent still exists, drop steps whose
// agent was deleted, append a fresh card for every pool agent not yet represented,
// then clean up edges that pointed at dropped steps. All three modes lay agents
// out radially by order, so new agents simply append (the Judge stays at index 0).
// Runs only when the pool changes (the effect depends on `agents`), so it never
// clobbers a connection the user just drew.
function reconcilePoolSteps(agents: Agent[], prev: WorkflowStep[]): WorkflowStep[] {
  const poolIds = new Set(agents.map((a) => a.id));
  const kept = prev.filter((s) => s.agentId && poolIds.has(s.agentId));
  const represented = new Set(kept.map((s) => s.agentId));
  const missing = agents.filter((a) => !represented.has(a.id));
  const next = [...kept, ...missing.map((a) => emptyStep({ agentId: a.id }))];

  const stepIds = new Set(next.map((s) => s.id));
  return next.map((s) => ({
    ...s,
    inputs: (s.inputs ?? []).filter((id) => stepIds.has(id)),
    judgeExcluded: (s.judgeExcluded ?? []).filter((id) => stepIds.has(id)),
  }));
}

// "?" icon next to a step heading — hover/focus shows what that step does.
function HelpHint({ text }: { text: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label="What is this step?"
            className="text-muted-foreground/70 hover:text-foreground transition-colors"
          >
            <HelpCircle className="h-[18px] w-[18px]" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-xs text-xs leading-relaxed">
          {text}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function AgentCard({
  agent,
  onConfigure,
  onDuplicate,
  onRemove,
  canRemove,
}: {
  agent: Agent;
  onConfigure: () => void;
  onDuplicate: () => void;
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
        className={`shrink-0 w-32 h-32 rounded-lg overflow-hidden bg-muted/40 flex items-center justify-center ${
          typeof agent.avatar === "number" ? "border" : "border border-dashed border-muted-foreground/40"
        }`}
      >
        {typeof agent.avatar === "number" ? (
          <div className="w-full h-full" style={avatarStyle(agent.avatar)} aria-hidden />
        ) : (
          <User className="h-14 w-14 text-muted-foreground/60" />
        )}
      </div>

      <div className="flex-1 min-w-0 flex flex-col gap-2">
        <div className="space-y-1 pr-6">
          <div className="text-sm font-semibold truncate">
            {agent.name ? (
              agent.name
            ) : (
              <span className="text-muted-foreground italic">Unnamed agent</span>
            )}
          </div>
          <div className="text-xs text-muted-foreground font-mono truncate">
            {providerLabel(agent.providerId)} / {agent.model || "—"}
          </div>
        </div>

        <div className="flex flex-wrap gap-1 text-[10px]">
          {agent.category && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Category: <strong className="font-semibold text-foreground">{agent.category}</strong></span>}
          {agent.personalityStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Personality: <strong className="font-semibold text-foreground">{agent.personalityStyle}</strong></span>}
          {agent.communicationStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Communication: <strong className="font-semibold text-foreground">{agent.communicationStyle}</strong></span>}
          {agent.responseStyle && <span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Response: <strong className="font-semibold text-foreground">{agent.responseStyle}</strong></span>}
        </div>

        <div className="mt-auto pt-1">
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="gap-1.5 text-xs" onClick={onConfigure}>
              <Settings2 className="h-3.5 w-3.5" /> Configure
            </Button>
            <Button variant="outline" size="sm" className="gap-1.5 text-xs" onClick={onDuplicate}>
              <Copy className="h-3.5 w-3.5" /> Duplicate
            </Button>
          </div>
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
  const [workflowMode, setWorkflowMode] = useSessionState<WorkflowMode | null>("agentpanel_mode", "reconcilier");
  useEffect(() => {
    // Judge (reconcilier) is the default mode: a missing/legacy-null stored value
    // is coerced to "reconcilier" so a mode is always selected.
    const migrated = migrateLegacyMode(workflowMode) ?? "reconcilier";
    if (migrated !== workflowMode) setWorkflowMode(migrated);
  }, [workflowMode, setWorkflowMode]);
  const [workflowSteps, setWorkflowSteps] = useSessionState<WorkflowStep[]>("agentpanel_steps", []);
  // Personalized mode: number of agent lines shown. Lines can be empty (three
  // "+" slots), which steps alone can't represent, so the count is tracked here.
  // "Add AI Agent line" increments it; the layout also renders enough lines to
  // cover all steps, so a stale/restored count never hides a populated line.
  const [personalizedLineCount, setPersonalizedLineCount] = useSessionState<number>("agentpanel_lineCount", 2);
  // Each template (mode) keeps its own step pool: switching modes stashes the
  // outgoing mode's steps here and restores the incoming mode's, so e.g.
  // Deliberation's 6 agents don't bleed into Manager. A mode absent from this map
  // has never been visited and is seeded to its own default on first switch.
  const [savedModeSteps, setSavedModeSteps] = useSessionState<Partial<Record<WorkflowMode, WorkflowStep[]>>>(
    "agentpanel_modeSteps",
    {},
  );
  // Which modes the user has manually edited. While a mode is NOT here, its
  // workflow mirrors the agent pool (see the mirror effect). Once edited, the
  // mode's layout is frozen and respected until Start Over.
  const [editedModes, setEditedModes] = useSessionState<Partial<Record<WorkflowMode, boolean>>>(
    "agentpanel_editedModes",
    {},
  );
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

  // No agent is seeded by default — the pool starts empty and the user adds
  // agents via the role buttons. The Configure Agents grid reserves a card-sized
  // empty slot so the section keeps its height when the pool is empty.
  const [hydrationDone, setHydrationDone] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setHydrationDone(true), 0);
    return () => clearTimeout(timer);
  }, []);

  // Every mode's cards are a live projection of the Configure Agents pool — one
  // card per agent (add/remove agents there, not on the canvas). Reconcile on each
  // pool change while preserving each surviving card's edits (step order, worker
  // connections, cut Judge links, per-card columns, persona). Runs after hydration
  // so it doesn't clobber persisted/restored state on first paint.
  useEffect(() => {
    if (!hydrationDone || !workflowMode) return;
    setWorkflowSteps((prev) => reconcilePoolSteps(agents, prev));
  }, [hydrationDone, workflowMode, agents, setWorkflowSteps]);

  const updateAgent = (id: string, updated: Agent) => {
    setAgents((prev) => prev.map((a) => (a.id === id ? updated : a)));
  };

  // Insert a copy right after the source card, with a fresh id and a unique name.
  // Naming: strip any existing "(copy)"/"(copyN)" suffix to get the base, then pick
  // the first free slot in the sequence "(copy)", "(copy1)", "(copy2)", … so duplicating
  // either the original or an existing copy never collides with a name already in the pool.
  const duplicateAgent = (id: string) => {
    const newId = makeAgentId();
    setAgents((prev) => {
      const idx = prev.findIndex((a) => a.id === id);
      if (idx === -1) return prev;
      const src = prev[idx];
      const existing = new Set(prev.map((a) => a.name));
      let name: string;
      if (src.name) {
        const base = src.name.replace(/ \(copy\d*\)$/, "");
        name = `${base} (copy)`;
        for (let n = 1; existing.has(name); n++) name = `${base} (copy${n})`;
      } else {
        name = `Agent ${prev.length + 1}`;
      }
      const copy = normalizeAgent({ ...src, id: newId, name });
      return [...prev.slice(0, idx + 1), copy, ...prev.slice(idx + 1)];
    });
  };
  const removeAgent = (id: string) => {
    setAgents((prev) => prev.filter((a) => a.id !== id));
    // Also clear any workflow step that references it — in the active pool and in
    // every other mode's stashed pool, so a restored mode doesn't show a dangling agent.
    setWorkflowSteps((prev) => prev.map((s) => (s.agentId === id ? { ...s, agentId: null } : s)));
    setSavedModeSteps((prev) => {
      const next: Partial<Record<WorkflowMode, WorkflowStep[]>> = {};
      for (const [mode, steps] of Object.entries(prev)) {
        next[mode as WorkflowMode] = (steps ?? []).map((s) => (s.agentId === id ? { ...s, agentId: null } : s));
      }
      return next;
    });
  };

  // Per-step live status during batch processing. Keyed by step.id.
  const markStepStatus = (id: string, status: StepStatus) => {
    setStepStatuses((prev) => ({ ...prev, [id]: status }));
  };

  // Mark the current mode's workflow as user-edited so the pool-mirror effect
  // stops overwriting it — the layout is then frozen until Start Over.
  const markWorkflowEdited = () => {
    if (workflowMode) {
      setEditedModes((prev) => (prev[workflowMode] ? prev : { ...prev, [workflowMode]: true }));
    }
  };

  const updateStep = (id: string, updated: WorkflowStep) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) => prev.map((s) => (s.id === id ? updated : s)));
  };
  // Removing a step also drops any edges that referenced it.
  const removeStep = (id: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) =>
      prev
        .filter((s) => s.id !== id)
        .map((s) =>
          (s.inputs ?? []).includes(id)
            ? { ...s, inputs: (s.inputs ?? []).filter((src) => src !== id) }
            : s,
        ),
    );
  };

  // Personalized mode: lines are visual rows; data flow is explicit edges only.
  // Add a fresh empty line (three "+" slots) by bumping the line count — no card
  // is added; the user fills the slots via the "+" buttons.
  const addAgentLine = () => {
    markWorkflowEdited();
    setPersonalizedLineCount((c) => c + 1);
  };
  const addStepToLine = (line: number, slot: number) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) => {
      const inLine = prev.filter((s) => (s.line ?? 0) === line);
      if (inLine.length >= 3) return prev;
      if (inLine.some((s) => (s.slot ?? 0) === slot)) return prev; // slot taken
      return [...prev, emptyStep({ line, slot })];
    });
  };
  // Configure Agents — a role button adds a preset agent to the pool, pre-filled
  // with a role-appropriate name, category, task, styles, and avatar (all fully
  // editable). Every mode's cards mirror the pool automatically (the reconcile
  // effect appends the new agent), so it surfaces with no further action.
  const addRoleAgent = (roleKey: string) => {
    const preset = AGENT_ROLE_PRESETS.find((p) => p.key === roleKey);
    if (!preset) return;
    const newId = makeAgentId();
    setAgents((prev) => {
      // Name agents after their role with a per-role counter — "Worker 1",
      // "Worker 2", … The number is one past the highest existing "<role> N" so
      // it never collides even after middle agents are removed. (role itself is
      // kept as an internal ordering hint for Judge/Sequential defaults; it's no
      // longer separately shown or edited.)
      const prefix = `${preset.label} `;
      const maxN = prev.reduce((m, a) => {
        if (!a.name.startsWith(prefix)) return m;
        const n = parseInt(a.name.slice(prefix.length), 10);
        return Number.isInteger(n) && n > m ? n : m;
      }, 0);
      const overrides: Partial<Agent> = {
        id: newId,
        name: `${preset.label} ${maxN + 1}`,
        role: preset.label,
        providerId: firstId,
        model: firstModel,
        category: preset.category,
        task: preset.task,
      };
      if (preset.personalityStyle) overrides.personalityStyle = preset.personalityStyle;
      if (preset.communicationStyle) overrides.communicationStyle = preset.communicationStyle;
      if (preset.responseStyle) overrides.responseStyle = preset.responseStyle;
      if (preset.personalityInstruction) overrides.personalityInstruction = preset.personalityInstruction;
      if (preset.dos) overrides.dos = [...preset.dos];
      if (preset.donts) overrides.donts = [...preset.donts];
      if (preset.avatar !== undefined) overrides.avatar = preset.avatar;
      return [...prev, emptyAgent(overrides)];
    });
  };
  // Edge from→to is stored as to.inputs containing from. Reject self-links,
  // duplicates, and anything that would introduce a cycle.
  const connectSteps = (fromId: string, toId: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) => {
      if (fromId === toId) return prev;
      const target = prev.find((s) => s.id === toId);
      if (!target) return prev;
      if ((target.inputs ?? []).includes(fromId)) return prev;
      if (wouldCreateCycle(prev, fromId, toId)) {
        toast.error("That connection would create a loop");
        return prev;
      }
      return prev.map((s) =>
        s.id === toId ? { ...s, inputs: [...(s.inputs ?? []), fromId] } : s,
      );
    });
  };
  const disconnectSteps = (fromId: string, toId: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) =>
      prev.map((s) =>
        s.id === toId
          ? { ...s, inputs: (s.inputs ?? []).filter((src) => src !== fromId) }
          : s,
      ),
    );
  };
  // Reconcilier mode — cut / restore a worker→Judge spoke. The set of cut
  // workers lives on the Judge step (steps[0]) as `judgeExcluded`; empty means
  // every worker feeds the Judge.
  const cutJudgeLink = (workerId: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) =>
      prev.map((s, i) =>
        i === 0
          ? { ...s, judgeExcluded: Array.from(new Set([...(s.judgeExcluded ?? []), workerId])) }
          : s,
      ),
    );
  };
  const restoreJudgeLink = (workerId: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) =>
      prev.map((s, i) =>
        i === 0
          ? { ...s, judgeExcluded: (s.judgeExcluded ?? []).filter((x) => x !== workerId) }
          : s,
      ),
    );
  };
  // Picking a different agent in a card's dropdown swaps that step's agent with the
  // step currently holding the picked agent, re-ordering without changing the agent
  // set. Used by both the Sequential steps and the Judge card (the Judge is just the
  // step at index 0, so it passes its own id). Only the agentId moves; each step
  // keeps its position-bound config (per-card columns, persona, prev-output toggle,
  // the Judge's judgeExcluded, worker→worker inputs).
  const swapStepAgents = (stepId: string, agentId: string) => {
    markWorkflowEdited();
    setWorkflowSteps((prev) => {
      const from = prev.find((s) => s.id === stepId);
      if (!from || from.agentId === agentId) return prev;
      const displaced = from.agentId;
      return prev.map((s) => {
        if (s.id === stepId) return { ...s, agentId };
        if (s.agentId === agentId) return { ...s, agentId: displaced };
        return s;
      });
    });
  };
  const handleModeChange = (m: WorkflowMode) => {
    if (m === workflowMode) return;
    // Stash the outgoing mode's steps so returning to it restores its own state.
    if (workflowMode) setSavedModeSteps((prev) => ({ ...prev, [workflowMode]: workflowSteps }));

    // If the user already shaped this mode's workflow, restore that exactly.
    // Otherwise default to one card per pool agent — the mirror effect then keeps
    // it in sync with the pool while it stays unedited.
    if (editedModes[m] && savedModeSteps[m] !== undefined) {
      setWorkflowSteps(savedModeSteps[m]!);
    } else {
      setWorkflowSteps(deriveStepsFromPool(m, agents));
    }
    // Drop the per-step run statuses so cards don't keep their green "done" ring
    // from a previous execution after the template changes.
    setStepStatuses({});
    setWorkflowMode(m);
  };

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
  // Display name for an uploaded document — used both in the per-card document
  // input block (executors) and the document chip label (workflow layout).
  const documentName = fileStates[0]?.file.name ?? "document";
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
    // Judge (reconcilier) and Personalized run with a single card. A lone Judge
    // makes a direct single-agent pass over the data; one worker + Judge runs the
    // worker then has the Judge synthesize it (both via the manual path in the
    // executor — no kappa, which is the only thing that needs ≥2 raters).
    if (workflowMode === "personalized" || workflowMode === "reconcilier") {
      if (workflowSteps.length < 1) return "Add at least one AI agent";
    } else if (workflowSteps.length < 2) {
      return "Add at least 2 workflow steps";
    }
    // Columns are optional: no selection → no columns are sent (only any per-card
    // extras). So we do NOT block the run on an empty selection.
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
  // No columns selected → nothing is sent (an empty `{}` / `[{}]`); columns are
  // never auto-filled. Per-card extras (the "+ column" picker) still flow through
  // the per-card content helpers below.
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

    // Per-card "Input columns" block — the card's kept selected columns for
    // this row (Personalized & Sequential honor per-card column removal).
    const colsMode = hasStructuredFile && outputFormat === "per-row";
    const cardColumnsBlock = (step: WorkflowStep): string => {
      if (!colsMode) return "";
      const inc = includedColumns(step, selectedCols, previewColumns);
      if (inc.length === 0) return "";
      const subset: Row = {};
      inc.forEach((c) => (subset[c] = row[c]));
      return `[Input columns]:\n${JSON.stringify(subset)}`;
    };

    // Per-card document block — the uploaded file's extracted text, included
    // unless the card opted out (the removable document chip). This lets a
    // connected card receive BOTH the document and its upstream output(s).
    const cardDocumentBlock = (step: WorkflowStep): string => {
      if (!hasUnstructuredFile || step.ignoreDocument) return "";
      return `[Document: ${documentName}]\n${userContent}`;
    };

    // Per-card model input — the card's own kept column subset (per-row mode),
    // or undefined to fall back to the shared `userContent` (non-structured /
    // instruction-only), or "" when the card opted out of the document. Shared
    // by the reconcilier workers/Judge and the deliberation agents.
    const cardUserContent = (step: WorkflowStep): string | undefined => {
      if (hasUnstructuredFile) return step.ignoreDocument ? "" : undefined;
      if (!colsMode) return undefined;
      // Exactly the columns shown on the card — `includedColumns` mirrors the
      // card's chip logic. Empty selection + no per-card extras → "{}" (no
      // columns), never the whole row.
      const inc = includedColumns(step, selectedCols, previewColumns);
      const subset: Row = {};
      inc.forEach((c) => (subset[c] = row[c]));
      return JSON.stringify(subset);
    };

    if (workflowMode === "sequential") {
      const outputs: Row = {};
      let totalLatencyMs = 0;
      let prevOutput: string | null = null;
      for (let i = 0; i < resolved.length; i++) {
        const { step, agent, prov } = resolved[i];
        const blocks: string[] = [];
        const cb = cardColumnsBlock(step);
        if (cb) blocks.push(cb);
        const db = cardDocumentBlock(step);
        if (db) blocks.push(db);
        if (i > 0 && !step.ignorePrevOutput && prevOutput != null) {
          blocks.push(`[From Step ${i}] Output:\n${prevOutput}`);
        }
        const stepInput = blocks.length > 0 ? blocks.join("\n\n") : userContent;

        markStepStatus(step.id, "running");
        const systemPrompt = `${aiInstructions}\n\n${composeStepSystemPrompt(agent, step)}`;
        try {
          const result = await dispatchProcessRow({
            provider: agent.providerId,
            model: agent.model,
            apiKey: prov?.apiKey ?? "",
            baseUrl: prov?.baseUrl,
            systemPrompt,
            userContent: stepInput,
            temperature: systemSettings.temperature,
            maxTokens: agent.maxTokens ?? systemSettings.maxTokens ?? undefined,
          });
          outputs[`step_${i + 1}_output`] = result.output;
          outputs[`step_${i + 1}_latency_ms`] = String(result.latency);
          totalLatencyMs += result.latency;
          prevOutput = result.output;
          markStepStatus(step.id, "done");
        } catch (err) {
          markStepStatus(step.id, "error");
          throw err;
        }
      }
      return { ...baseRow, ...outputs, status: "success", latency_ms: totalLatencyMs };
    }

    if (workflowMode === "personalized") {
      // Explicit DAG: each step's input is the concatenation of its connected
      // sources' outputs (labeled). A step with no incoming edges receives the
      // original row/file input. Execute in topological order.
      const labels = buildStepLabels(workflowSteps);
      const resolvedById = new Map(resolved.map((r) => [r.step.id, r]));
      const order = topoOrder(workflowSteps);
      const outputById = new Map<string, string>();

      // Per-agent system prompt = short output rule + the user's own AI
      // Instructions extras (text after the marker) + the step's composed
      // persona. The auto workflow preamble (orchestrator line, STEPS list,
      // WORKFLOW CONFIG json) is intentionally NOT sent to each agent.
      const outputRule =
        "Return only the final result — no explanation, no markdown, no code fences.";
      const markerIdx = aiInstructions.indexOf(AI_INSTRUCTIONS_MARKER);
      const userExtras =
        markerIdx === -1
          ? ""
          : aiInstructions.slice(markerIdx + AI_INSTRUCTIONS_MARKER.length).trim();

      const outputs: Row = {};
      let totalLatencyMs = 0;
      for (const stepId of order) {
        const entry = resolvedById.get(stepId);
        if (!entry) continue;
        const { step, agent, prov } = entry;
        const sources = (step.inputs ?? []).filter((s) => resolvedById.has(s));
        const blocks: string[] = [];
        const cb = cardColumnsBlock(step);
        if (cb) blocks.push(cb);
        const db = cardDocumentBlock(step);
        if (db) blocks.push(db);
        for (const src of sources) {
          const srcStep = resolvedById.get(src)?.step;
          const task = srcStep?.taskDescription?.trim();
          blocks.push(
            `[From ${labels[src] ?? src}]\n` +
              `Task: ${task || "(no specific task)"}\n` +
              `Output:\n${outputById.get(src) ?? ""}`,
          );
        }
        const stepInput = blocks.length > 0 ? blocks.join("\n\n") : userContent;

        markStepStatus(step.id, "running");
        const systemPrompt = [outputRule, userExtras, composeStepSystemPrompt(agent, step)]
          .filter(Boolean)
          .join("\n\n");
        try {
          const result = await dispatchProcessRow({
            provider: agent.providerId,
            model: agent.model,
            apiKey: prov?.apiKey ?? "",
            baseUrl: prov?.baseUrl,
            systemPrompt,
            userContent: stepInput,
            temperature: systemSettings.temperature,
            maxTokens: agent.maxTokens ?? systemSettings.maxTokens ?? undefined,
          });
          outputById.set(step.id, result.output);
          const colBase = (labels[step.id] ?? step.id)
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "");
          outputs[`${colBase}_output`] = result.output;
          outputs[`${colBase}_latency_ms`] = String(result.latency);
          totalLatencyMs += result.latency;
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
      // By default the Judge receives every worker's output, but the user can cut
      // individual worker→Judge spokes (tracked in `judgeExcluded` below). Workers
      // may also be wired to each other (worker→worker edges on their `inputs`): a
      // downstream worker receives its upstream workers' outputs and runs after
      // them. No edges + all connected → classic parallel-workers consensus;
      // otherwise → an ordered worker run whose connected outputs the Judge synthesizes.
      const workerStepIds = new Set(workers.map(({ step }) => step.id));
      const hasWorkerEdges = workers.some(({ step }) =>
        (step.inputs ?? []).some((id) => workerStepIds.has(id)),
      );
      // Worker→Judge spokes the user has cut (stored on the Judge step). The
      // Judge synthesizes only the workers still connected; the rest still RUN
      // (their output columns appear) but aren't fed to the Judge. `judgeWorkers`
      // keeps each worker's original index for stable "Worker N" labels.
      const judgeExcluded = new Set(reconciler.step.judgeExcluded ?? []);
      const judgeWorkers = workers
        .map((w, i) => ({ ...w, i }))
        .filter((w) => !judgeExcluded.has(w.step.id));
      const allConnected = judgeWorkers.length === workers.length;
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
      const perWorkerInstructions = judgeWorkers
        .map((w) => {
          const task = w.step.taskDescription.trim();
          return `worker_${w.i + 1} instruction: ${task || "(no specific instruction)"}`;
        })
        .join("\n");
      const reconcilerOnlyContext = `Per-worker tasks (each worker received only its own; this is your reference for judging compliance):\n${perWorkerInstructions}`;
      const reconcilerPrompt = userExtras
        ? `${reconcilerOnlyContext}\n\n${userExtras}`
        : reconcilerOnlyContext;
      const outputRule =
        "Return only the final result — no explanation, no markdown, no code fences.";

      // ── Judge-only (no workers) ─────────────────────────────────────────
      // A lone Judge card has nothing to synthesize, so it just runs directly on
      // the row data — a single-agent pass. Produces only judge_output. (No
      // per-worker context here: there are no workers.)
      if (workers.length === 0) {
        markStepStatus(reconciler.step.id, "running");
        const judgeSystem = [outputRule, userExtras, composeStepSystemPrompt(reconciler.agent, reconciler.step)]
          .filter(Boolean)
          .join("\n\n");
        try {
          const jr = await dispatchProcessRow({
            provider: reconciler.agent.providerId,
            model: reconciler.agent.model,
            apiKey: reconciler.prov?.apiKey ?? "",
            baseUrl: reconciler.prov?.baseUrl,
            systemPrompt: judgeSystem,
            userContent: cardUserContent(reconciler.step) ?? userContent,
            temperature: systemSettings.temperature,
            maxTokens: reconciler.agent.maxTokens ?? systemSettings.maxTokens ?? undefined,
          });
          markStepStatus(reconciler.step.id, "done");
          return {
            ...baseRow,
            judge_output: jr.output,
            judge_latency_ms: jr.latency,
            status: "success",
            latency_ms: jr.latency,
          };
        } catch (err) {
          markStepStatus(reconciler.step.id, "error");
          throw err;
        }
      }

      // ── Manual worker run → Judge synthesis ─────────────────────────────
      // Used whenever the graph isn't a clean parallel consensus: a single worker
      // (consensus/kappa needs ≥2 raters), worker→worker edges exist, OR some
      // worker→Judge spokes were cut. Workers run in topological order (a worker's
      // input is its own data plus every upstream worker's labeled output); the
      // Judge then synthesizes the still-connected workers' outputs in one call.
      // Not independent raters → no kappa here.
      if (hasWorkerEdges || !allConnected || workers.length < 2) {
        const order = topoOrder(workers.map(({ step }) => step));
        const byId = new Map(workers.map((w) => [w.step.id, w]));
        const indexById = new Map(workers.map((w, i) => [w.step.id, i]));
        const outputById = new Map<string, string>();
        const outputs: Row = {};
        const latencyCols: Row = {};

        workers.forEach(({ step }) => markStepStatus(step.id, "pending"));
        markStepStatus(reconciler.step.id, "pending");

        let totalWorkerMs = 0;
        for (const stepId of order) {
          const entry = byId.get(stepId);
          if (!entry) continue;
          const { step, agent, prov } = entry;
          const i = indexById.get(stepId) ?? 0;
          const own = cardUserContent(step);
          const blocks: string[] = [];
          if (own) blocks.push(own);
          for (const src of step.inputs ?? []) {
            if (!workerStepIds.has(src)) continue;
            const si = indexById.get(src) ?? 0;
            const srcTask = byId.get(src)?.step.taskDescription?.trim();
            blocks.push(
              `[From Worker ${si + 1}]\n` +
                `Task: ${srcTask || "(no specific task)"}\n` +
                `Output:\n${outputById.get(src) ?? ""}`,
            );
          }
          const stepInput =
            blocks.length > 0 ? blocks.join("\n\n") : own === "" ? "" : userContent;

          markStepStatus(step.id, "running");
          const systemPrompt = [outputRule, userExtras, composeStepSystemPrompt(agent, step)]
            .filter(Boolean)
            .join("\n\n");
          try {
            const r = await dispatchProcessRow({
              provider: agent.providerId,
              model: agent.model,
              apiKey: prov?.apiKey ?? "",
              baseUrl: prov?.baseUrl,
              systemPrompt,
              userContent: stepInput,
              temperature: systemSettings.temperature,
              maxTokens: agent.maxTokens ?? systemSettings.maxTokens ?? undefined,
            });
            outputById.set(step.id, r.output);
            outputs[`worker_${i + 1}_output`] = r.output;
            latencyCols[`worker_${i + 1}_latency_ms`] = r.latency;
            totalWorkerMs += r.latency;
            markStepStatus(step.id, "done");
          } catch (err) {
            markStepStatus(step.id, "error");
            throw err;
          }
        }

        // Judge synthesizes the still-connected workers' outputs, labeled by
        // their original card number.
        markStepStatus(reconciler.step.id, "running");
        const judgeBlocks = judgeWorkers.map((w) => {
          const task = w.step.taskDescription.trim();
          return `[Worker ${w.i + 1}]${task ? ` (task: ${task})` : ""}\n${outputById.get(w.step.id) ?? ""}`;
        });
        const judgeInput = `Worker outputs to synthesize:\n\n${judgeBlocks.join("\n\n")}`;
        const judgeSystem = [outputRule, reconcilerPrompt, composeStepSystemPrompt(reconciler.agent, reconciler.step)]
          .filter(Boolean)
          .join("\n\n");
        let judgeMs = 0;
        try {
          const jr = await dispatchProcessRow({
            provider: reconciler.agent.providerId,
            model: reconciler.agent.model,
            apiKey: reconciler.prov?.apiKey ?? "",
            baseUrl: reconciler.prov?.baseUrl,
            systemPrompt: judgeSystem,
            userContent: judgeInput,
            temperature: systemSettings.temperature,
            maxTokens: reconciler.agent.maxTokens ?? systemSettings.maxTokens ?? undefined,
          });
          outputs["judge_output"] = jr.output;
          latencyCols["judge_latency_ms"] = jr.latency;
          judgeMs = jr.latency;
          markStepStatus(reconciler.step.id, "done");
        } catch (err) {
          markStepStatus(reconciler.step.id, "error");
          throw err;
        }
        return {
          ...baseRow,
          ...outputs,
          ...latencyCols,
          status: "success",
          latency_ms: totalWorkerMs + judgeMs,
        };
      }

      // ── No worker→worker edges: classic parallel-workers consensus ──────
      // All workers kick off in parallel inside the dispatch — mark them together.
      workers.forEach(({ step }) => markStepStatus(step.id, "running"));
      markStepStatus(reconciler.step.id, "pending");
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
          userContent: cardUserContent(step),
        })),
        reconciler: {
          provider: reconciler.agent.providerId,
          model: reconciler.agent.model,
          apiKey: reconciler.prov?.apiKey ?? "",
          baseUrl: reconciler.prov?.baseUrl,
          persona: composeStepSystemPrompt(reconciler.agent, reconciler.step),
        },
        reconcilerUserContent: cardUserContent(reconciler.step),
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
      outputs["judge_output"] = result.reconcilerOutput;
      if (typeof result.reconcilerLatency === "number" && Number.isFinite(result.reconcilerLatency)) {
        latencyCols["judge_latency_ms"] = Math.round(result.reconcilerLatency * 1000);
      }
      // Wall-clock per row: workers run in parallel, then reconciler runs sequentially.
      // Latencies from ConsensusResult are in seconds; convert to ms.
      const workerLatencies = (result.workerResults ?? []).map((w) => w.latency);
      const maxWorker = workerLatencies.length > 0 ? Math.max(...workerLatencies) : 0;
      const latencyMs = Math.round((maxWorker + (result.reconcilerLatency ?? 0)) * 1000);
      return { ...baseRow, ...outputs, ...latencyCols, status: "success", latency_ms: latencyMs };
    }

    // deliberation — each agent's round-1 input is its own kept column subset
    // (via the shared cardUserContent); undefined falls back to userContent.
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
        userContent: cardUserContent(step),
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
    // `reconciler_output` retained for older saved runs; new runs use `judge_output`.
    const FINAL_KEYS = new Set(["final_output", "final_consensus", "judge_output", "reconciler_output", "rounds_taken", "converged"]);
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

      const restoredMode = cfg.mode !== undefined ? migrateLegacyMode(cfg.mode) : workflowMode;
      if (cfg.mode !== undefined) setWorkflowMode(restoredMode);
      if (Array.isArray(cfg.agents) && cfg.agents.length > 0) {
        setAgents(cfg.agents.map((a) => normalizeAgent(a)));
      }
      if (Array.isArray(cfg.steps)) {
        setWorkflowSteps(cfg.steps);
        // Freeze the restored mode so the pool-mirror effect doesn't overwrite
        // the saved layout with a fresh pool-derived default.
        if (restoredMode) setEditedModes((prev) => ({ ...prev, [restoredMode]: true }));
      }
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
    // clearSessionKeys only wipes sessionStorage; the in-memory column selection
    // survives (and its auto-reset branch can't fire once previewColumns is empty),
    // so clear it explicitly or the workflow cards keep showing the old data columns.
    setSelectedCols([]);
    setAgents([]);
    setWorkflowMode("reconcilier");
    // Empty pool → empty workflow; the mirror effect repopulates cards as the
    // user adds agents (until they manually edit the workflow).
    setWorkflowSteps([]);
    setPersonalizedLineCount(2);
    setSavedModeSteps({});
    setEditedModes({});
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
          <h1 className="text-4xl font-bold">Multi-Agent Workflows</h1>
          <p className="text-muted-foreground text-sm">Unified multi-agent workflows — Judge, Sequential, or Deliberation.</p>
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
              description="Choose which columns to send into the workflow for each row. Optional — if none are selected, no columns are sent (a card can still pull in specific columns via its own “+ column”)."
            />
          </div>
        </>
      )}

      <div className="border-t" />

      {/* ── N. Configure Agents */}
      <div className="space-y-4 py-8">
        <div className="flex items-center gap-2">
          <h2 className="text-2xl font-bold">{nAgents}. Configure Agents</h2>
          <HelpHint text="Build your pool of AI agents. For each agent pick its provider and model, and set its role, personality, and instructions. Agents defined here can be assigned to one or more steps in the workflow below, and saved to the shared library to reuse on other pages." />
        </div>
        <p className="text-xs text-muted-foreground">
          Define your agent pool — assign them to steps below.
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
              {agents.length === 0 ? (
                // Empty pool — reserve one card's worth of height so the section
                // doesn't collapse to just the buttons.
                <div className="min-h-[8.5rem]" aria-hidden />
              ) : (
                agents.map((a) => (
                  <AgentCard
                    key={a.id}
                    agent={a}
                    onConfigure={() => setConfiguringId(a.id)}
                    onDuplicate={() => duplicateAgent(a.id)}
                    onRemove={() => removeAgent(a.id)}
                    canRemove
                  />
                ))
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {AGENT_ROLE_PRESETS.map((preset) => (
                <Button
                  key={preset.key}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                  title={preset.task}
                  onClick={() => addRoleAgent(preset.key)}
                >
                  <Plus className="h-3.5 w-3.5 mr-1.5" /> Add {preset.buttonLabel ?? preset.label}
                </Button>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="border-t" />

      {/* ── N. Configure Template */}
      <div className="space-y-4 py-8">
        <div className="flex items-center gap-2">
          <h2 className="text-2xl font-bold">{nMode}. Configure Template</h2>
          <HelpHint text="Choose how the agents collaborate. Personalized: build your own custom lines and connections. Sequential: a pipeline where each step's output feeds the next. Judge: workers run in parallel and a judge merges their answers. Deliberation: all agents discuss over several rounds. This sets the structure used in the Agent Workflow step." />
        </div>
        <WorkflowModeSelector value={workflowMode} onChange={handleModeChange} />
      </div>

      {workflowMode && (
        <>
          <div className="border-t" />

          {/* ── N. Agent Workflow */}
          <div className="space-y-4 py-8">
            <div className="flex items-center gap-2">
              <h2 className="text-2xl font-bold">{nWorkflow}. Agent Workflow</h2>
              <HelpHint text="Assign an agent to each step and give it its task and input data. The layout follows the template chosen above." />
            </div>
            <p className="text-xs text-muted-foreground">
              {workflowMode === "sequential" && "Steps run in order — each feeds the next."}
              {workflowMode === "reconcilier" && "Workers run in parallel; the top judge card synthesizes their outputs."}
              {workflowMode === "deliberation" && "Agents deliberate over rounds, each seeing the others' outputs."}
              {workflowMode === "personalized" && "Agents come from Configure Agents. Connect them: click a card’s “Connect” button, then another agent. Runs in dependency order."}
            </p>
            <WorkflowLayout
              mode={workflowMode}
              steps={workflowSteps}
              agents={agents}
              stepStatuses={stepStatuses}
              onUpdate={updateStep}
              onRemove={removeStep}
              onAddLine={addAgentLine}
              onAddToLine={addStepToLine}
              onConnect={connectSteps}
              onDisconnect={disconnectSteps}
              onCutJudge={cutJudgeLink}
              onRestoreJudge={restoreJudgeLink}
              onSwapStepAgent={swapStepAgents}
              selectedCols={selectedCols}
              allCols={previewColumns}
              documentInput={hasUnstructuredFile ? documentName : undefined}
              lineCount={personalizedLineCount}
            />
          </div>

          <div className="border-t" />

          {hasStructuredFile && nOutputFormat && (
            <>
              <div className="space-y-4 py-8">
                <h2 className="text-2xl font-bold">{nOutputFormat}. Output Format</h2>
                <p className="text-xs text-muted-foreground">
                  How should the agents handle your file — row by row, or all rows at once?
                </p>
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
          existingNames={agents.filter((a) => a.id !== configuringAgent.id).map((a) => a.name)}
        />
      )}
    </div>
  );
}
