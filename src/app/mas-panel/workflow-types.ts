import { type Agent, buildAgentSystemPrefix } from "@/lib/agent-library";

export type WorkflowMode = "reconcilier" | "sequential" | "deliberation" | "personalized";

export const STEP_MINIMUMS: Record<WorkflowMode, number> = {
  reconcilier: 4,
  // Sequential starts with two steps; the trailing "+" card adds more on demand.
  sequential: 2,
  deliberation: 5,
  // Personalized starts empty — the user builds lines by hand.
  personalized: 0,
};

/** Back-compat: an earlier release used `"wizard"` for what is now `"reconcilier"`. */
export function migrateLegacyMode(m: WorkflowMode | null | string): WorkflowMode | null {
  if (m === "wizard") return "reconcilier";
  return m as WorkflowMode | null;
}

export interface WorkflowStep {
  id: string;
  agentId: string | null;        // References an Agent in the step-2 pool
  taskDescription: string;       // Step-specific task override
  persona: string;               // Additive persona on top of the agent's own
  additionalKnowledge: string;   // Additive knowledge on top of the agent's own
  /**
   * Personalized mode only — which row ("line") this step is grouped under
   * visually. Lines are organizational only; data flow is defined entirely by
   * explicit `inputs` connections. Other modes ignore this (absent → line 0).
   */
  line?: number;
  /**
   * Personalized mode only — column position within its line (0-based). Slots
   * may have holes (e.g. agent at slot 0 and slot 2, slot 1 empty).
   */
  slot?: number;
  /**
   * Personalized mode only — ids of steps whose outputs feed this step. An edge
   * A→B is stored as B.inputs containing A.id. A step with no inputs receives
   * the original row/file input. The graph must stay acyclic.
   */
  inputs?: string[];
  /**
   * Personalized & Sequential — selected columns the user removed from THIS
   * card's Input DATA. Card input uses (global selected columns) minus these.
   */
  excludedCols?: string[];
  /**
   * Sequential only — when true this step does NOT receive the previous step's
   * output (the removable "Step N-1 output" chip). Default false.
   */
  ignorePrevOutput?: boolean;
  /**
   * Unstructured file (PDF/DOCX/TXT) input only — when true this step does NOT
   * receive the uploaded document's extracted text (the removable document
   * chip). Lets a card rely solely on connected/previous outputs. Default false.
   */
  ignoreDocument?: boolean;
}

/** Columns this card actually feeds the model = global selected minus excluded. */
export function includedColumns(step: WorkflowStep, allCols: string[]): string[] {
  const ex = new Set(step.excludedCols ?? []);
  return allCols.filter((c) => !ex.has(c));
}

export interface DeliberationSettings {
  maxRounds: number;
  convergenceMode: "fixed" | "adaptive";
  convergenceThreshold: number;  // 0-50, only used when convergenceMode === "adaptive"
}

export const DEFAULT_DELIBERATION_SETTINGS: DeliberationSettings = {
  maxRounds: 3,
  convergenceMode: "fixed",
  convergenceThreshold: 10,
};

export function makeStepId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function emptyStep(overrides: Partial<WorkflowStep> = {}): WorkflowStep {
  return {
    id: makeStepId(),
    agentId: null,
    taskDescription: "",
    persona: "",
    additionalKnowledge: "",
    line: 0,
    slot: 0,
    inputs: [],
    excludedCols: [],
    ignorePrevOutput: false,
    ignoreDocument: false,
    ...overrides,
  };
}

/** Max agent columns per line in Personalized mode. */
export const MAX_AGENTS_PER_LINE = 3;

/** Group steps into lines, sorted by ascending `line` value. */
export function groupLines(steps: WorkflowStep[]): [number, WorkflowStep[]][] {
  const groups = new Map<number, WorkflowStep[]>();
  for (const s of steps) {
    const ln = s.line ?? 0;
    if (!groups.has(ln)) groups.set(ln, []);
    groups.get(ln)!.push(s);
  }
  return [...groups.entries()].sort((a, b) => a[0] - b[0]);
}

/**
 * Place a line's steps into a fixed array of MAX_AGENTS_PER_LINE columns by
 * their `slot`. Holes (undefined) are allowed. Collisions or out-of-range slots
 * fall back to the first free column so malformed/legacy data still renders.
 */
export function placeLineSteps(
  lineSteps: WorkflowStep[],
): (WorkflowStep | undefined)[] {
  const placed: (WorkflowStep | undefined)[] = new Array(MAX_AGENTS_PER_LINE).fill(undefined);
  const ordered = [...lineSteps].sort((a, b) => (a.slot ?? 0) - (b.slot ?? 0));
  for (const s of ordered) {
    let pos = Math.min(Math.max(s.slot ?? 0, 0), MAX_AGENTS_PER_LINE - 1);
    if (placed[pos]) {
      pos = placed.findIndex((x) => !x);
      if (pos === -1) continue;
    }
    placed[pos] = s;
  }
  return placed;
}

/**
 * Compose the full system prompt for a workflow step.
 * Order: agent's base prefix → the step's PRIMARY TASK (emphasized, placed
 * first so the model treats it as the main objective) → supporting step
 * persona → supporting step knowledge. Per-step values layer on top of the
 * agent's own; they never replace them.
 *
 * Pass `includeTask: false` when the caller is forwarding the task via a separate
 * channel (e.g. Reconcilier passes `taskDescription` to the consensus dispatcher's
 * `instruction`/`reconcilerPrompt` field) to avoid duplicating it in the system prompt.
 */
export function composeStepSystemPrompt(
  agent: Agent,
  step: WorkflowStep,
  opts: { includeTask?: boolean } = {},
): string {
  const { includeTask = true } = opts;
  const parts: string[] = [];

  const base = buildAgentSystemPrefix(agent);
  if (base) parts.push(base);

  // Skip the PRIMARY TASK block when the step task merely repeats the agent's
  // main goal (already emitted by buildAgentSystemPrefix above) — e.g. the Main
  // Prompt box was seeded from the agent and left unedited — so the goal isn't
  // sent twice.
  const taskText = step.taskDescription.trim();
  if (includeTask && taskText && taskText !== (agent.goal ?? "").trim()) {
    parts.push("");
    parts.push("════ PRIMARY TASK — your main objective for this step ════");
    parts.push(taskText);
    parts.push(
      "Focus on completing this task above all else. The sections below are supporting context only.",
    );
  }

  if (step.persona.trim()) {
    parts.push("");
    parts.push("Supporting persona for this step:");
    parts.push(step.persona.trim());
  }

  if (step.additionalKnowledge.trim()) {
    parts.push("");
    parts.push("Supporting knowledge for this step:");
    parts.push(step.additionalKnowledge.trim());
  }

  return parts.join("\n");
}

// ── Personalized-mode DAG helpers ────────────────────────────────────────────

/**
 * Human-readable label for a step in Personalized mode, e.g. "Agent 4".
 * Agents are numbered sequentially across the whole grid (MAX_AGENTS_PER_LINE per
 * line) by their fixed line/slot position: Line 1 → Agent 1‑3, Line 2 → Agent 4‑6,
 * and so on. Positional numbering keeps a card's label stable even when other
 * cards or whole lines are left empty.
 */
export function buildStepLabels(steps: WorkflowStep[]): Record<string, string> {
  const labels: Record<string, string> = {};
  groupLines(steps).forEach(([lineNo, lineSteps]) => {
    placeLineSteps(lineSteps).forEach((s, pos) => {
      if (s) labels[s.id] = `Agent ${lineNo * MAX_AGENTS_PER_LINE + pos + 1}`;
    });
  });
  return labels;
}

/**
 * True if adding edge from→to would create a cycle, i.e. `to` can already reach
 * `from` by following inputs backwards (edge A→B ⇒ B.inputs has A).
 */
export function wouldCreateCycle(
  steps: WorkflowStep[],
  fromId: string,
  toId: string,
): boolean {
  if (fromId === toId) return true;
  const byId = new Map(steps.map((s) => [s.id, s]));
  // Walk *upstream* from `from`: its inputs, their inputs, … If we reach `to`,
  // then to→…→from already exists and from→to would close a loop.
  const stack = [fromId];
  const seen = new Set<string>();
  while (stack.length) {
    const cur = stack.pop()!;
    if (cur === toId) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const src of byId.get(cur)?.inputs ?? []) stack.push(src);
  }
  return false;
}

/**
 * Kahn topological order over the inputs graph. Returns step ids in an order
 * where every step appears after all of its `inputs`. Any steps left in a cycle
 * (shouldn't happen — creation prevents cycles) are appended at the end.
 */
export function topoOrder(steps: WorkflowStep[]): string[] {
  const ids = new Set(steps.map((s) => s.id));
  const byId = new Map(steps.map((s) => [s.id, s]));
  const indegree = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  for (const s of steps) {
    indegree.set(s.id, 0);
    dependents.set(s.id, []);
  }
  for (const s of steps) {
    for (const src of s.inputs ?? []) {
      if (!ids.has(src)) continue; // dangling edge — ignore
      indegree.set(s.id, (indegree.get(s.id) ?? 0) + 1);
      dependents.get(src)!.push(s.id);
    }
  }
  const queue = steps.filter((s) => (indegree.get(s.id) ?? 0) === 0).map((s) => s.id);
  const order: string[] = [];
  while (queue.length) {
    const cur = queue.shift()!;
    order.push(cur);
    for (const dep of dependents.get(cur) ?? []) {
      indegree.set(dep, (indegree.get(dep) ?? 1) - 1);
      if ((indegree.get(dep) ?? 0) === 0) queue.push(dep);
    }
  }
  // Append any unresolved (cyclic) nodes so execution still covers them.
  for (const s of steps) if (!order.includes(s.id)) order.push(s.id);
  return order.filter((id) => byId.has(id));
}
