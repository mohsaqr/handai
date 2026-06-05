// Shared agent library — persisted in localStorage, usable across tool pages.
import type { CSSProperties } from "react";

export interface Agent {
  id: string;
  name: string;
  /** Short role label shown alongside the name (e.g. "Agent 1 (Reviewer)"). Defaults to "Agent". */
  role: string;
  providerId: string;
  model: string;
  category: string;
  /** Main system prompt — the agent's primary goal (e.g. "You help students with their homework…"). */
  goal: string;
  personalityStyle: string;
  personalityInstruction: string;
  communicationStyle: string;
  responseStyle: string;
  knowledgeContext: string;
  /** Per-agent max response length in tokens. Null = use the global default. */
  maxTokens: number | null;
  /** Strict rules — behaviors to encourage. */
  dos: string[];
  /** Strict rules — behaviors to avoid. */
  donts: string[];
  /** Index into the avatars sprite sheet (public/avatars.png). 0..31. */
  avatar?: number;
}

// Sprite-sheet metadata for public/avatars.png.
// Source sheet is 8 columns × 4 rows = 32 cells, indexed left-to-right, top-to-bottom.
export const AGENT_AVATAR_SHEET_COLS = 8;
export const AGENT_AVATAR_SHEET_ROWS = 4;

// We only expose the right half of the sheet — the 4 rightmost columns of each row
// (16 avatars total). Stored values are sheet indices (0..31), so existing data stays valid.
export const AGENT_AVATAR_INDICES = [
  4, 5, 6, 7,
  12, 13, 14, 15,
  20, 21, 22, 23,
  28, 29, 30, 31,
] as const;
export const AGENT_AVATAR_COUNT = AGENT_AVATAR_INDICES.length;

const AVATAR_SHEET_URL = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/avatars.png`;

/** Returns inline CSS that crops one cell of public/avatars.png by sheet index (0..31). */
export function avatarStyle(index: number): CSSProperties {
  const col = index % AGENT_AVATAR_SHEET_COLS;
  const row = Math.floor(index / AGENT_AVATAR_SHEET_COLS);
  return {
    backgroundImage: `url(${AVATAR_SHEET_URL})`,
    backgroundSize: `${AGENT_AVATAR_SHEET_COLS * 100}% ${AGENT_AVATAR_SHEET_ROWS * 100}%`,
    backgroundPosition: `${(col / (AGENT_AVATAR_SHEET_COLS - 1)) * 100}% ${(row / (AGENT_AVATAR_SHEET_ROWS - 1)) * 100}%`,
    backgroundRepeat: "no-repeat",
  };
}

export const AGENT_CATEGORIES = [
  "Neutral",
  "Analyst",
  "Critic",
  "Creative",
  "Synthesizer",
  "Researcher",
  "Devil's Advocate",
  "Specialist",
] as const;

export const PERSONALITY_STYLES = [
  "Neutral",
  "Formal",
  "Concise",
  "Verbose",
  "Technical",
  "Empathetic",
  "Direct",
  "Diplomatic",
] as const;

// Reused from agent-network tool.
export const COMMUNICATION_STYLES = [
  "Neutral",
  "Collaborative",
  "Adversarial",
  "Deliberative",
  "Socratic",
] as const;

export const RESPONSE_STYLES = [
  "Concise",
  "Balanced",
  "Detailed",
] as const;

const STORAGE_KEY = "handai_agent_library";

export function makeAgentId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function emptyAgent(overrides: Partial<Agent> = {}): Agent {
  return {
    id: makeAgentId(),
    name: "",
    role: "Agent",
    providerId: "openai",
    model: "gpt-4o",
    category: AGENT_CATEGORIES[0],
    goal: "",
    personalityStyle: PERSONALITY_STYLES[0],
    personalityInstruction: "",
    communicationStyle: COMMUNICATION_STYLES[0],
    responseStyle: RESPONSE_STYLES[1],
    knowledgeContext: "",
    maxTokens: null,
    dos: [],
    donts: [],
    ...overrides,
  };
}

/**
 * Fills any fields missing from a persisted/restored agent with current
 * defaults. Library entries and historical-run snapshots saved before a field
 * was introduced lack it — without this they'd drive uncontrolled inputs and
 * skip the new behavior. `emptyAgent` spreads defaults then the stored values.
 */
export function normalizeAgent(a: Partial<Agent>): Agent {
  return emptyAgent(a);
}

function readStore(): Agent[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map((a) => normalizeAgent(a)) : [];
  } catch {
    return [];
  }
}

function writeStore(agents: Agent[]): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(agents));
    window.dispatchEvent(new CustomEvent("handai-agent-library-changed"));
  } catch {
    /* quota or serialization failure — silently drop */
  }
}

export function listAgents(): Agent[] {
  return readStore();
}

export function saveAgent(agent: Agent): void {
  const agents = readStore();
  const idx = agents.findIndex((a) => a.id === agent.id);
  if (idx >= 0) agents[idx] = agent;
  else agents.push(agent);
  writeStore(agents);
}

export function deleteAgent(id: string): void {
  writeStore(readStore().filter((a) => a.id !== id));
}

export function loadAgent(id: string): Agent | null {
  return readStore().find((a) => a.id === id) ?? null;
}

// Build the per-agent system prompt that will be prepended to the shared base prompt.
export function buildAgentSystemPrefix(agent: Agent): string {
  const parts: string[] = [];
  if (agent.name) parts.push(`You are an agent named "${agent.name}".`);
  if (agent.category && agent.category !== "Neutral")
    parts.push(`Your role category is: ${agent.category}.`);
  if (agent.goal?.trim()) {
    parts.push("");
    parts.push("Main goal:");
    parts.push(agent.goal.trim());
    parts.push("");
  }
  if (agent.personalityStyle && agent.personalityStyle !== "Neutral")
    parts.push(`Respond in a ${agent.personalityStyle.toLowerCase()} tone.`);
  if (agent.communicationStyle && agent.communicationStyle !== "Neutral")
    parts.push(`Use a ${agent.communicationStyle.toLowerCase()} communication style.`);
  if (agent.responseStyle) {
    const rs = agent.responseStyle.toLowerCase();
    const guide = rs === "concise"
      ? "Keep answers brief and to the point."
      : rs === "detailed"
      ? "Provide thorough, in-depth answers with supporting detail."
      : "Give balanced answers — neither overly brief nor overly long.";
    parts.push(guide);
  }
  if (agent.personalityInstruction.trim()) {
    parts.push("");
    parts.push("Personality description:");
    parts.push(agent.personalityInstruction.trim());
  }
  const dos = (agent.dos ?? []).map((r) => r.trim()).filter(Boolean);
  const donts = (agent.donts ?? []).map((r) => r.trim()).filter(Boolean);
  if (dos.length > 0 || donts.length > 0) {
    parts.push("");
    parts.push("Strict rules:");
    dos.forEach((r) => parts.push(`- DO: ${r}`));
    donts.forEach((r) => parts.push(`- DON'T: ${r}`));
  }
  if (agent.knowledgeContext.trim()) {
    parts.push("");
    parts.push("Additional knowledge context (use as reference when answering):");
    parts.push(agent.knowledgeContext.trim());
  }
  return parts.join("\n");
}
