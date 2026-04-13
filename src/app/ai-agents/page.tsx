"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
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
import { Plus, X, RotateCcw, ChevronDown, ChevronRight, Shield } from "lucide-react";
import { toast } from "sonner";
import { dispatchAgentsRow } from "@/lib/llm-dispatch";
import { SmartFileUpload, type FileStatus } from "@/components/tools/SmartFileUpload";
import { SAMPLE_DATASETS, sampleAsFile } from "@/lib/sample-data";
import { parseStructuredFile } from "@/lib/parse-file";
import { extractTextBrowser } from "@/lib/document-browser";
import { AIInstructionsSection } from "@/components/tools/AIInstructionsSection";
import { useAIInstructions, AI_INSTRUCTIONS_MARKER } from "@/hooks/useAIInstructions";
import { useSessionState, clearSessionKeys } from "@/hooks/useSessionState";
import { ExecutionPanel } from "@/components/tools/ExecutionPanel";
import { ResultsPanel } from "@/components/tools/ResultsPanel";
import { getPrompt } from "@/lib/prompts";

type Row = Record<string, unknown>;
type DataMode = "structured" | "unstructured";

// ── Agent config types ───────────────────────────────────────────────────────

interface AgentConfig {
  id: string;
  name: string;
  rolePreset: string;
  role: string;
  providerId: string;
  model: string;
  selectedColumns: string[];
  isReferee: boolean;
}

const AGENT_ROLE_PRESETS: Record<string, { label: string; promptId: string }> = {
  critic:          { label: "Critic",          promptId: "agents.critic" },
  defender:        { label: "Defender",        promptId: "agents.defender" },
  synthesizer:     { label: "Synthesizer",     promptId: "agents.synthesizer" },
  domain_expert:   { label: "Domain Expert",   promptId: "agents.domain_expert" },
  devils_advocate: { label: "Devil's Advocate", promptId: "agents.devils_advocate" },
  mediator:        { label: "Mediator",        promptId: "agents.mediator" },
  referee:         { label: "Referee",         promptId: "agents.referee" },
  custom:          { label: "Custom",          promptId: "" },
};

function providerLabel(id: string) {
  if (id === "lmstudio") return "LM Studio";
  if (id === "ollama") return "Ollama";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

function makeId() {
  return Math.random().toString(36).slice(2, 10);
}

// ── Agent Card ──────────────────────────────────────────────────────────────

function AgentCard({
  agent,
  onUpdate,
  onRemove,
  canRemove,
  enabledProviders,
  allColumns,
  isStructured,
}: {
  agent: AgentConfig;
  onUpdate: (a: AgentConfig) => void;
  onRemove: () => void;
  canRemove: boolean;
  enabledProviders: { providerId: string; defaultModel: string; isEnabled: boolean; apiKey?: string; baseUrl?: string; isLocal?: boolean }[];
  allColumns: string[];
  isStructured: boolean;
}) {
  const [showPrompt, setShowPrompt] = useState(false);

  const handlePresetChange = (preset: string) => {
    const def = AGENT_ROLE_PRESETS[preset];
    const role = def?.promptId ? getPrompt(def.promptId) : agent.role;
    onUpdate({ ...agent, rolePreset: preset, role });
  };

  const handleRefereeToggle = () => {
    onUpdate({ ...agent, isReferee: !agent.isReferee });
  };

  const toggleColumn = (col: string) => {
    const cols = agent.selectedColumns.includes(col)
      ? agent.selectedColumns.filter((c) => c !== col)
      : [...agent.selectedColumns, col];
    onUpdate({ ...agent, selectedColumns: cols });
  };

  const toggleAllColumns = () => {
    const allSelected = allColumns.every((c) => agent.selectedColumns.includes(c));
    onUpdate({ ...agent, selectedColumns: allSelected ? [] : [...allColumns] });
  };

  return (
    <div className={`border rounded-lg p-4 space-y-3 relative ${agent.isReferee ? "border-amber-400 bg-amber-50/30 dark:bg-amber-950/20" : ""}`}>
      {canRemove && (
        <button onClick={onRemove} className="absolute top-3 right-3 text-muted-foreground hover:text-destructive" title="Remove agent">
          <X className="h-4 w-4" />
        </button>
      )}

      {/* Name + Referee */}
      <div className="flex items-center gap-3">
        <Input
          value={agent.name}
          onChange={(e) => onUpdate({ ...agent, name: e.target.value })}
          className="text-sm font-semibold flex-1"
          placeholder="Agent name"
        />
        <button
          onClick={handleRefereeToggle}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium border transition-colors ${
            agent.isReferee
              ? "bg-amber-100 border-amber-300 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 dark:border-amber-700"
              : "bg-muted/50 border-border text-muted-foreground hover:bg-muted"
          }`}
          title={agent.isReferee ? "This agent is the referee" : "Make this agent the referee"}
        >
          <Shield className="h-3.5 w-3.5" />
          Referee
        </button>
      </div>

      {/* Role Preset + Provider/Model */}
      <div className="grid grid-cols-3 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Role</Label>
          <Select value={agent.rolePreset} onValueChange={handlePresetChange}>
            <SelectTrigger className="text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(AGENT_ROLE_PRESETS).map(([key, { label }]) => (
                <SelectItem key={key} value={key} className="text-xs">{label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Provider</Label>
          <Select value={agent.providerId} onValueChange={(v) => onUpdate({ ...agent, providerId: v })}>
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
                <SelectItem value={agent.providerId} className="text-xs">No providers</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Model</Label>
          <Input
            value={agent.model}
            onChange={(e) => onUpdate({ ...agent, model: e.target.value })}
            placeholder="e.g. gpt-4o"
            className="text-xs font-mono"
          />
        </div>
      </div>

      {/* Persona/Prompt (collapsible) */}
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
            value={agent.role}
            onChange={(e) => onUpdate({ ...agent, role: e.target.value, rolePreset: "custom" })}
            className="mt-2 min-h-[120px] text-xs font-mono resize-y"
          />
        )}
      </div>

      {/* Per-agent column selection (structured only, non-referee) */}
      {isStructured && !agent.isReferee && allColumns.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground">Columns this agent sees</Label>
            <button onClick={toggleAllColumns} className="text-xs text-blue-600 hover:underline">
              {allColumns.every((c) => agent.selectedColumns.includes(c)) ? "Deselect all" : "Select all"}
            </button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {allColumns.map((col) => (
              <label key={col} className="flex items-center gap-1 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={agent.selectedColumns.includes(col)}
                  onChange={() => toggleColumn(col)}
                  className="accent-primary w-3 h-3"
                />
                <span className="truncate max-w-[120px]">{col}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Negotiation Log Viewer ──────────────────────────────────────────────────

function NegotiationLogViewer({ logJson }: { logJson: string }) {
  let log: Array<{ round: number; agent: string; output: string }> = [];
  try { log = JSON.parse(logJson); } catch { return null; }
  if (!log.length) return null;

  const rounds = new Map<number, Array<{ agent: string; output: string }>>();
  for (const entry of log) {
    if (!rounds.has(entry.round)) rounds.set(entry.round, []);
    rounds.get(entry.round)!.push({ agent: entry.agent, output: entry.output });
  }

  return (
    <div className="space-y-2">
      {Array.from(rounds.entries()).map(([round, entries]) => (
        <details key={round} className="border rounded p-2">
          <summary className="text-xs font-medium cursor-pointer">
            Round {round} ({entries.length} agents)
          </summary>
          <div className="mt-2 space-y-2">
            {entries.map((e, i) => (
              <div key={i} className="bg-muted/30 rounded p-2">
                <div className="text-xs font-semibold text-muted-foreground mb-1">{e.agent}</div>
                <div className="text-xs whitespace-pre-wrap">{e.output}</div>
              </div>
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

export default function AIAgentsPage() {
  const providers = useAppStore((state) => state.providers);
  const systemSettings = useSystemSettings();
  const enabledProviders = Object.values(providers).filter((p) => p.isEnabled);
  const firstId = enabledProviders[0]?.providerId ?? "openai";
  const firstModel = enabledProviders[0]?.defaultModel ?? "gpt-4o";
  const secondId = enabledProviders[1]?.providerId ?? firstId;
  const secondModel = enabledProviders[1]?.defaultModel ?? firstModel;

  // ── State ────────────────────────────────────────────────────────────────
  const [data, setData] = useSessionState<Row[]>("aiagents_data", []);
  const [dataName, setDataName] = useSessionState("aiagents_dataName", "");
  const [dataMode, setDataMode] = useSessionState<DataMode>("aiagents_dataMode", "structured");
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [maxRounds, setMaxRounds] = useSessionState("aiagents_maxRounds", 3);

  const makeDefaultAgents = useCallback((): AgentConfig[] => [
    { id: makeId(), name: "Analyst", rolePreset: "domain_expert", role: getPrompt("agents.domain_expert"), providerId: firstId, model: firstModel, selectedColumns: [], isReferee: false },
    { id: makeId(), name: "Critic", rolePreset: "critic", role: getPrompt("agents.critic"), providerId: secondId, model: secondModel, selectedColumns: [], isReferee: false },
    { id: makeId(), name: "Referee", rolePreset: "referee", role: getPrompt("agents.referee"), providerId: firstId, model: firstModel, selectedColumns: [], isReferee: true },
  ], [firstId, firstModel, secondId, secondModel]);

  const [agents, setAgents] = useSessionState<AgentConfig[]>("aiagents_agents", []);

  // Initialize default agents after hydration
  const initRef = useRef(false);
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    if (agents.length === 0 && enabledProviders.length > 0) {
      setAgents(makeDefaultAgents());
    }
  }, [agents.length, enabledProviders.length, makeDefaultAgents, setAgents]);

  // ── Selected log row for expanded view ──────────────────────────────────
  const [expandedLogIdx, setExpandedLogIdx] = useState<number | null>(null);

  const allColumns = data.length > 0 ? Object.keys(data[0]) : [];

  // Auto-select all columns for new agents when data changes
  useEffect(() => {
    if (dataMode !== "structured" || allColumns.length === 0) return;
    setAgents((prev) =>
      prev.map((a) =>
        a.selectedColumns.length === 0 && !a.isReferee
          ? { ...a, selectedColumns: [...allColumns] }
          : a
      )
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allColumns.join(","), dataMode]);

  // ── Agent CRUD ──────────────────────────────────────────────────────────
  const updateAgent = (id: string, updated: AgentConfig) => {
    setAgents((prev) => {
      // If this agent is being set as referee, unset all others
      if (updated.isReferee) {
        return prev.map((a) =>
          a.id === id ? updated : { ...a, isReferee: false }
        );
      }
      return prev.map((a) => (a.id === id ? updated : a));
    });
  };

  const addAgent = () => {
    setAgents((prev) => [
      ...prev,
      {
        id: makeId(),
        name: `Agent ${prev.length + 1}`,
        rolePreset: "custom",
        role: "",
        providerId: firstId,
        model: firstModel,
        selectedColumns: dataMode === "structured" ? [...allColumns] : [],
        isReferee: false,
      },
    ]);
  };

  const removeAgent = (id: string) => {
    setAgents((prev) => prev.filter((a) => a.id !== id));
  };

  // ── AI Instructions ─────────────────────────────────────────────────────
  const buildAutoInstructions = useCallback(() => {
    const lines: string[] = [];
    lines.push("Multi-agent negotiation: agents analyze data over multiple rounds, then a referee produces the final answer.");
    lines.push("");
    lines.push(`Max rounds: ${maxRounds}`);
    lines.push("");
    const nonRefs = agents.filter((a) => !a.isReferee);
    const ref = agents.find((a) => a.isReferee);
    if (nonRefs.length > 0) {
      lines.push("AGENTS:");
      nonRefs.forEach((a) => {
        const preset = AGENT_ROLE_PRESETS[a.rolePreset]?.label ?? "Custom";
        lines.push(`- ${a.name} (${preset}): ${a.providerId}/${a.model}`);
      });
      lines.push("");
    }
    if (ref) {
      lines.push(`REFEREE: ${ref.name} (${ref.providerId}/${ref.model})`);
      lines.push("");
    }
    lines.push(AI_INSTRUCTIONS_MARKER);
    return lines.join("\n");
  }, [agents, maxRounds]);

  const [aiInstructions, setAiInstructions] = useAIInstructions(buildAutoInstructions);

  // ── Active model for batch processor (use referee) ──────────────────────
  const referee = agents.find((a) => a.isReferee);
  const refereeProvider = referee ? providers[referee.providerId] : null;
  const activeModel = refereeProvider && referee ? {
    ...refereeProvider,
    providerId: referee.providerId,
    defaultModel: referee.model,
  } : null;

  // ── Batch processor ─────────────────────────────────────────────────────
  const batch = useBatchProcessor({
    toolId: "/ai-agents",
    runType: "ai-agents",
    activeModel,
    systemSettings,
    data,
    dataName,
    systemPrompt: aiInstructions,
    validate: () => {
      const nonRefs = agents.filter((a) => !a.isReferee);
      const refs = agents.filter((a) => a.isReferee);
      if (nonRefs.length < 2) return "Need at least 2 non-referee agents";
      if (refs.length !== 1) return "Exactly one agent must be the referee";
      for (const agent of agents) {
        const prov = providers[agent.providerId];
        if (!prov) return `Invalid provider for "${agent.name}"`;
        if (!prov.isLocal && !prov.apiKey) return `API key missing for "${agent.name}". Check Settings.`;
      }
      if (dataMode === "structured") {
        for (const agent of nonRefs) {
          if (agent.selectedColumns.length === 0) return `"${agent.name}" has no columns selected`;
        }
      }
      return null;
    },
    runParams: {
      provider: referee?.providerId ?? firstId,
      model: referee?.model ?? firstModel,
      temperature: systemSettings.temperature,
    },
    processRow: async (row: Row, idx: number) => {
      const agentConfigs = agents.map((agent) => {
        const prov = providers[agent.providerId];
        return {
          name: agent.name,
          role: agent.role,
          provider: agent.providerId,
          model: agent.model,
          apiKey: prov?.apiKey || "local",
          baseUrl: prov?.baseUrl,
          columns: dataMode === "structured" && !agent.isReferee ? agent.selectedColumns : undefined,
          isReferee: agent.isReferee,
        };
      });

      const userContent = dataMode === "unstructured"
        ? (row.document_text as string) || JSON.stringify(row)
        : JSON.stringify(row);

      const result = await dispatchAgentsRow({
        agents: agentConfigs,
        userContent,
        maxRounds,
        rowIdx: idx,
      });

      const agentCols: Record<string, string> = {};
      result.agentOutputs.forEach((ao) => {
        agentCols[`agent_${ao.name}_output`] = ao.output;
      });

      return {
        ...row,
        ...agentCols,
        referee_output: result.refereeOutput,
        negotiation_log: JSON.stringify(result.negotiationLog),
        rounds_taken: result.roundsTaken,
        converged: result.converged ? "yes" : "no",
        status: "success",
      };
    },
    buildResultEntry: (r: Row, i: number) => ({
      rowIndex: i,
      input: r as Record<string, unknown>,
      output: (r.referee_output ?? "") as string,
      status: (r.status === "error" ? "error" : "success") as string,
      errorMessage: r.error_msg as string | undefined,
    }),
  });

  // ── Session restore ─────────────────────────────────────────────────────
  const restored = useRestoreSession("ai-agents");
  React.useEffect(() => {
    if (!restored) return;
    queueMicrotask(() => {
      setData(restored.data as Row[]);
      setDataName(restored.dataName);
      const errors = restored.results.filter((r) => r.status === "error").length;
      useProcessingStore.getState().completeJob(
        "/ai-agents",
        restored.results as Row[],
        { success: restored.results.length - errors, errors, avgLatency: 0 },
        restored.runId,
      );
      toast.success(`Restored session from "${restored.dataName}" (${restored.data.length} rows)`);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restored]);

  const adoptStructured = useCallback((file: File, rows: Row[]) => {
    setCurrentFile(file);
    setData(rows);
    setDataName(file.name);
    setDataMode("structured");
    batch.clearResults();
    setExpandedLogIdx(null);
  }, [batch, setData, setDataName, setDataMode]);

  const adoptUnstructured = useCallback((file: File, text: string) => {
    setCurrentFile(file);
    setData([{ document_text: text, file_name: file.name }]);
    setDataName(file.name);
    setDataMode("unstructured");
    batch.clearResults();
    setExpandedLogIdx(null);
  }, [batch, setData, setDataName, setDataMode]);

  const handleDrop = useCallback(async (accepted: File[]) => {
    const file = accepted[0];
    if (!file) return;
    const rows = await parseStructuredFile(file);
    if (rows && rows.length > 0) {
      adoptStructured(file, rows as Row[]);
      toast.success(`Loaded ${rows.length} rows from ${file.name}`);
      return;
    }
    try {
      const { text, charCount, truncated } = await extractTextBrowser(file);
      if (!text.trim()) {
        toast.error("Document appears to be empty or unreadable");
        return;
      }
      adoptUnstructured(file, text);
      toast.success(`Loaded document: ${file.name} (${charCount.toLocaleString()} chars${truncated ? ", truncated" : ""})`);
    } catch (err) {
      toast.error(`Failed to extract text: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [adoptStructured, adoptUnstructured]);

  const handleLoadSample = useCallback((key: string) => {
    const made = sampleAsFile(key);
    if (!made) return;
    adoptStructured(made.file, made.rows as Row[]);
    toast.success(`Loaded sample: ${SAMPLE_DATASETS[key].name}`);
  }, [adoptStructured]);

  const handleClearFile = useCallback(() => {
    setCurrentFile(null);
    setData([]);
    setDataName("");
    setDataMode("structured");
    batch.clearResults();
    setExpandedLogIdx(null);
  }, [batch, setData, setDataName, setDataMode]);

  const nonReferees = agents.filter((a) => !a.isReferee);
  const callsPerRow = nonReferees.length * maxRounds + 1;

  const handleStartOver = () => {
    clearSessionKeys("aiagents_");
    setCurrentFile(null);
    setData([]);
    setDataName("");
    setDataMode("structured");
    setAgents(makeDefaultAgents());
    setMaxRounds(3);
    setAiInstructions("");
    setExpandedLogIdx(null);
    batch.clearResults();
  };

  const resultRow = batch.results[0];
  let fileStatus: FileStatus = "pending";
  if (batch.isProcessing) fileStatus = "processing";
  else if (resultRow?.status === "error") fileStatus = "error";
  else if (resultRow?.status === "success") fileStatus = "done";
  const fileError = resultRow?.error_msg as string | undefined;

  return (
    <div className="space-y-0 pb-16">

      {/* Header */}
      <div className="pb-6 flex items-start justify-between">
        <div className="space-y-1 max-w-3xl">
          <h1 className="text-4xl font-bold">AI Agents</h1>
          <p className="text-muted-foreground text-sm">
            Multi-agent negotiation with role-based personas and iterative refinement
          </p>
        </div>
        <Button variant="destructive" className="gap-2 px-5" onClick={handleStartOver}>
            <RotateCcw className="h-3.5 w-3.5" /> Start Over
          </Button>
      </div>

      <div className={batch.isProcessing ? "pointer-events-none opacity-60" : ""}>

      {/* ── 1. Upload Data ──────────────────────────────────────────────────── */}
      <div className="space-y-4 pb-8">
        <h2 className="text-2xl font-bold">1. Upload Data</h2>
        <SmartFileUpload
          file={currentFile}
          status={fileStatus}
          errorMessage={fileError}
          previewRows={dataMode === "structured" ? data : null}
          onDrop={handleDrop}
          onClear={handleClearFile}
          onLoadSample={handleLoadSample}
        />
      </div>

      <div className="border-t" />

      {/* ── 2. Configure Agents ─────────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">2. Configure Agents</h2>
          <Button variant="outline" size="sm" className="text-xs" onClick={addAgent}>
            <Plus className="h-3.5 w-3.5 mr-1.5" /> Add Agent
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          Each agent has a persona and sees specific columns. One agent must be the referee (produces the final answer).
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {agents.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              onUpdate={(updated) => updateAgent(agent.id, updated)}
              onRemove={() => removeAgent(agent.id)}
              canRemove={agents.length > 3}
              enabledProviders={enabledProviders}
              allColumns={allColumns}
              isStructured={dataMode === "structured"}

            />
          ))}
        </div>
      </div>

      <div className="border-t" />

      {/* ── 3. Negotiation Settings ─────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">3. Negotiation Settings</h2>
        <div className="flex items-center gap-6">
          <div className="space-y-2">
            <Label className="text-sm">Max Rounds</Label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1}
                max={10}
                value={maxRounds}
                onChange={(e) => setMaxRounds(Number(e.target.value))}
                className="w-48 accent-primary"
              />
              <span className="text-sm font-mono w-6 text-center">{maxRounds}</span>
            </div>
          </div>
          <div className="text-xs text-muted-foreground max-w-md">
            Agents refine their outputs over {maxRounds} round{maxRounds > 1 ? "s" : ""}. Each round, agents see all other agents&apos; previous outputs.
            Stops early if all agents converge (same output as previous round).
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          Estimated cost: ~{callsPerRow} API calls per row ({nonReferees.length} agents x {maxRounds} round{maxRounds > 1 ? "s" : ""} + 1 referee call)
        </div>
      </div>

      <div className="border-t" />

      {/* ── 4. AI Instructions ──────────────────────────────────────────────── */}
      <AIInstructionsSection
        sectionNumber={4}
        value={aiInstructions}
        onChange={setAiInstructions}
      />

      </div>

      <div className="border-t" />

      {/* ── 5. Execute ──────────────────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">5. Execute</h2>
        <ExecutionPanel
          isProcessing={batch.isProcessing}
          aborting={batch.aborting}
          runMode={batch.runMode}
          progress={batch.progress}
          etaStr={batch.etaStr}
          dataCount={data.length}
          disabled={data.length === 0 || agents.length < 3}
          onRun={batch.run}
          onAbort={batch.abort}
          onResume={batch.resume}
          onCancel={batch.clearResults}
          failedCount={batch.failedCount}
          skippedCount={batch.skippedCount}
        />
      </div>

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      <ResultsPanel
        results={batch.results}
        runId={batch.runId}
        title="Results"
        subtitle={`${batch.results.length} rows processed`}
      >
        {/* Expandable negotiation log for selected row */}
        {batch.results.length > 0 && (
          <div className="space-y-3">
            <div className="text-sm font-medium">Negotiation Log</div>
            <div className="flex flex-wrap gap-1.5">
              {batch.results.map((row, idx) => (
                <button
                  key={idx}
                  onClick={() => setExpandedLogIdx(expandedLogIdx === idx ? null : idx)}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${
                    expandedLogIdx === idx
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-muted/50 border-border hover:bg-muted"
                  }`}
                >
                  Row {idx + 1}
                </button>
              ))}
            </div>
            {expandedLogIdx !== null && batch.results[expandedLogIdx] && (
              <div className="border rounded-lg p-3">
                <div className="text-xs font-medium mb-2">
                  Row {expandedLogIdx + 1} — {String(batch.results[expandedLogIdx].rounds_taken)} round(s), {batch.results[expandedLogIdx].converged === "yes" ? "converged" : "did not converge"}
                </div>
                <NegotiationLogViewer logJson={(batch.results[expandedLogIdx].negotiation_log as string) || "[]"} />
                <div className="mt-3 p-2 bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded">
                  <div className="text-xs font-semibold text-amber-800 dark:text-amber-300 mb-1">Referee Output</div>
                  <div className="text-xs whitespace-pre-wrap">{batch.results[expandedLogIdx].referee_output as string}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </ResultsPanel>
    </div>
  );
}
