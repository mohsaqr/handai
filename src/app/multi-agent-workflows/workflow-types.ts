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
   * Default (selected) columns the user removed from THIS card's Input DATA.
   * Card input = (global selected columns − excludedCols) + extraCols.
   */
  excludedCols?: string[];
  /**
   * Uploaded columns the user added to THIS card beyond the global selection
   * (via the "+ column" picker). These are NOT in the global selected set —
   * they let a card pull in any column from the uploaded file, not just the
   * globally-defined ones.
   */
  extraCols?: string[];
  /**
   * Reconcilier mode only — stored on the JUDGE step (`steps[0]`). Worker ids
   * whose output the user has cut from the Judge (scissored that worker→Judge
   * spoke). Absent/empty → every worker feeds the Judge (the default); new
   * workers auto-connect since they aren't listed here.
   */
  judgeExcluded?: string[];
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

/**
 * Columns this card actually feeds the model: the global selected columns minus
 * the card's excluded ones, plus any extra columns the card pulled in from the
 * uploaded file. Ordered by the uploaded-file column order (`allCols`).
 */
export function includedColumns(
  step: WorkflowStep,
  selectedCols: string[],
  allCols: string[],
): string[] {
  const excluded = new Set(step.excludedCols ?? []);
  const extra = new Set(step.extraCols ?? []);
  const selected = new Set(selectedCols);
  return allCols.filter((c) => extra.has(c) || (selected.has(c) && !excluded.has(c)));
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
    extraCols: [],
    ignorePrevOutput: false,
    ignoreDocument: false,
    ...overrides,
  };
}

/** Max agent columns per line in Personalized mode. */
export const MAX_AGENTS_PER_LINE = 3;

/**
 * Configure Agents — preset role buttons. Each adds a pre-filled (but fully
 * editable) agent to the pool. `category`/`personalityStyle`/`communicationStyle`/
 * `responseStyle` must match the option sets in agent-library.ts; `avatar` is a
 * sprite index from AGENT_AVATAR_INDICES. These are templates only — they do not
 * change execution; the user still assigns agents to steps and draws connections.
 */
export interface RolePreset {
  key: string;
  label: string;
  /** Button text override. Defaults to `label`. The agent's name/role still use
   *  `label` (e.g. "Worker 1"), so this only affects the button caption. */
  buttonLabel?: string;
  category: string;
  task: string;
  personalityStyle?: string;
  communicationStyle?: string;
  responseStyle?: string;
  /** Free-text personality description applied to the agent. */
  personalityInstruction?: string;
  /** Behaviours to encourage / avoid — fill the agent's strict rules. */
  dos?: string[];
  donts?: string[];
  avatar?: number;
}

export const AGENT_ROLE_PRESETS: RolePreset[] = [
  {
    key: "worker",
    label: "Worker",
    buttonLabel: "Worker (neutral)",
    category: "Neutral",
    task: "",
    personalityStyle: "Neutral",
    communicationStyle: "Neutral",
    responseStyle: "Balanced",
    avatar: 4,
  },
  {
    key: "judge",
    label: "Judge",
    category: "Critic",
    task: "",
    personalityStyle: "Formal",
    communicationStyle: "Deliberative",
    responseStyle: "Detailed",
    personalityInstruction:
      "You are an impartial adjudicator. You weigh each input on its merits, justify your verdict, and stay fair and consistent across cases.",
    dos: [
      "Compare inputs against the explicit task criteria",
      "Explain the reasoning behind your verdict",
      "Select or synthesize the single strongest answer",
    ],
    donts: [
      "Favor an input without justification",
      "Introduce requirements that aren't in the task",
    ],
    avatar: 14,
  },
  {
    key: "manager",
    label: "Manager",
    category: "Synthesizer",
    task: "",
    personalityStyle: "Direct",
    communicationStyle: "Collaborative",
    responseStyle: "Balanced",
    personalityInstruction:
      "You are a decisive coordinator. You decompose work into clear subtasks, keep the end goal in focus, and integrate results into one coherent deliverable.",
    dos: [
      "Break the task into clear, ordered subtasks",
      "Integrate the inputs into a single coherent output",
      "Resolve conflicts between inputs before finalizing",
    ],
    donts: [
      "Duplicate work across subtasks",
      "Leave conflicting outputs unresolved",
    ],
    avatar: 6,
  },
  {
    key: "critic",
    label: "Critic",
    category: "Critic",
    task: "",
    personalityStyle: "Direct",
    communicationStyle: "Adversarial",
    responseStyle: "Detailed",
    personalityInstruction:
      "You are a rigorous reviewer. You probe for weaknesses, edge cases, and unsupported claims, and you are precise about what is wrong and why.",
    dos: [
      "Point to specific flaws, risks, and gaps",
      "Challenge unsupported assumptions",
      "Suggest how each issue could be fixed",
    ],
    donts: [
      "Give vague or generic praise",
      "Approve the input without scrutiny",
    ],
    avatar: 7,
  },
  {
    key: "researcher",
    label: "Researcher",
    category: "Researcher",
    task: "",
    personalityStyle: "Technical",
    communicationStyle: "Neutral",
    responseStyle: "Balanced",
    personalityInstruction:
      "You are a thorough investigator. You collect the relevant facts, organize them clearly, and carefully separate evidence from inference.",
    dos: [
      "Lay out the relevant facts and context",
      "Distinguish established facts from assumptions",
      "Note gaps or uncertainty in the evidence",
    ],
    donts: [
      "State speculation as fact",
      "Omit sources of uncertainty",
    ],
    avatar: 30,
  },
  {
    key: "synthesizer",
    label: "Synthesizer",
    category: "Synthesizer",
    task: "",
    personalityStyle: "Concise",
    communicationStyle: "Collaborative",
    responseStyle: "Balanced",
    personalityInstruction:
      "You are an integrator. You merge multiple inputs into one clear whole, preserving every key point while removing duplication and contradiction.",
    dos: [
      "Combine the inputs into one coherent output",
      "Preserve every distinct key point",
      "Remove duplication and reconcile contradictions",
    ],
    donts: [
      "Simply concatenate the inputs",
      "Drop important details while merging",
    ],
    avatar: 28,
  },
];

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
  // main task (already emitted by buildAgentSystemPrefix above) — e.g. the Main
  // Prompt box was seeded from the agent and left unedited — so the task isn't
  // sent twice.
  const taskText = step.taskDescription.trim();
  if (includeTask && taskText && taskText !== (agent.task ?? "").trim()) {
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
