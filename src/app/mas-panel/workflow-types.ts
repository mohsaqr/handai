import { type Agent, buildAgentSystemPrefix } from "@/lib/agent-library";

export type WorkflowMode = "reconcilier" | "sequential" | "deliberation";

export const STEP_MINIMUMS: Record<WorkflowMode, number> = {
  reconcilier: 4,
  sequential: 3,
  deliberation: 6,
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
    ...overrides,
  };
}

/**
 * Compose the full system prompt for a workflow step.
 * Additive: agent's base prefix + step persona + step knowledge + step task.
 * The per-step values never *replace* the agent's own — they layer on top.
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

  if (step.persona.trim()) {
    parts.push("");
    parts.push("Additional persona for this step:");
    parts.push(step.persona.trim());
  }

  if (step.additionalKnowledge.trim()) {
    parts.push("");
    parts.push("Additional knowledge for this step:");
    parts.push(step.additionalKnowledge.trim());
  }

  if (includeTask && step.taskDescription.trim()) {
    parts.push("");
    parts.push("Task for this step:");
    parts.push(step.taskDescription.trim());
  }

  return parts.join("\n");
}
