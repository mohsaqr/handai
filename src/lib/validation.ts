import { z } from 'zod';

// ── Shared sub-schemas ─────────────────────────────────────────────────────────

const ProviderFieldsLocal = z.object({
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(''),
  baseUrl: z.string().optional(),
});

const LlmTuningFields = {
  temperature: z.number().min(0).max(2).optional(),
  maxTokens: z.number().int().positive().optional(),
};

// ── /api/process-row ──────────────────────────────────────────────────────────
export const ProcessRowSchema = z.object({
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  systemPrompt: z.string(),
  userContent: z.string(),
  rowIdx: z.number().int().optional(),
  runId: z.string().optional(),
  ...LlmTuningFields,
});

// ── /api/consensus-row ────────────────────────────────────────────────────────
const ConsensusWorkerFields = ProviderFieldsLocal.extend({
  persona: z.string().optional(),
  // Per-worker input override (Manager template per-card column removal).
  // Falls back to the shared `userContent` when absent.
  userContent: z.string().optional(),
});

export const ConsensusRowSchema = z.object({
  workers: z.array(ConsensusWorkerFields).min(2),
  reconciler: ConsensusWorkerFields,
  workerPrompt: z.string(),
  reconcilerPrompt: z.string(),
  userContent: z.string(),
  // The "Original Data" the reconciler/manager sees. Falls back to
  // `userContent` when absent (per-card column removal on the manager card).
  reconcilerUserContent: z.string().optional(),
  rowIdx: z.number().int().optional(),
  runId: z.string().optional(),
  enableQualityScoring: z.boolean().optional(),
  enableDisagreementAnalysis: z.boolean().optional(),
  includeReasoning: z.boolean().optional(),
  ...LlmTuningFields,
});

// ── /api/agent-network-row ──────────────────────────────────────────────────
const NetworkAgentSchema = z.object({
  label: z.string().min(1),
  role: z.string().default(""),
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  // Per-agent input override (the card's own column subset / document opt-out).
  // Falls back to the shared `userContent` when omitted.
  userContent: z.string().optional(),
});

export const AgentNetworkRowSchema = z.object({
  agents: z.array(NetworkAgentSchema).min(2),
  userContent: z.string(),
  maxRounds: z.number().int().min(1).max(10).default(3),
  communicationStyle: z.string().optional(),
  convergenceMode: z.enum(["fixed", "adaptive"]).optional(),
  convergenceThreshold: z.number().min(0).max(100).optional(),
  rowIdx: z.number().int().optional(),
  runId: z.string().optional(),
  ...LlmTuningFields,
});

// ── /api/automator-row ────────────────────────────────────────────────────────
export const AutomatorRowSchema = z.object({
  row: z.record(z.string(), z.unknown()),
  steps: z.array(
    z.object({
      name: z.string(),
      task: z.string(),
      input_fields: z.array(z.string()),
      output_fields: z.array(
        z.object({
          name: z.string(),
          type: z.string(),
          constraints: z.string().optional(),
        })
      ),
    })
  ).min(1),
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  ...LlmTuningFields,
});

// ── /api/generate-row ─────────────────────────────────────────────────────────
export const GenerateRowSchema = z.object({
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  systemPrompt: z.string().optional(),
  rowCount: z.number().int().min(1).max(500),
  columns: z.array(
    z.object({
      name: z.string(),
      type: z.enum(['text', 'number', 'boolean', 'list']),
      description: z.string().optional(),
    })
  ).optional(),
  freeformPrompt: z.string().optional(),
  outputFormat: z.enum(["tabular", "json", "freetext", "markdown", "gift"]).optional(),
  ...LlmTuningFields,
});

// ── Document shared sub-schemas ───────────────────────────────────────────────

const FieldDefSchema = z.object({
  name: z.string(),
  type: z.enum(['text', 'number', 'date', 'boolean', 'list']),
  description: z.string(),
});

const DocumentFileTypeEnum = z.enum(['pdf', 'docx', 'excel', 'txt', 'md', 'json', 'html', 'csv']);

// ── /api/document-extract ─────────────────────────────────────────────────────
// When `structuredRows` is provided, `fileContent` is ignored and the rows are
// sent to the LLM as a JSON payload (better than re-serializing them as text).
export const DocumentExtractSchema = z.object({
  fileContent: z.string().default(""),
  fileType: DocumentFileTypeEnum,
  fileName: z.string().optional(),
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  systemPrompt: z.string().optional(),
  fields: z.array(FieldDefSchema).optional(),
  structuredRows: z.array(z.record(z.string(), z.unknown())).optional(),
  ...LlmTuningFields,
}).refine(
  (v) => v.fileContent.length > 0 || (v.structuredRows && v.structuredRows.length > 0),
  { message: "Either fileContent or structuredRows must be provided", path: ["fileContent"] }
);

// ── /api/document-analyze ─────────────────────────────────────────────────────
export const DocumentAnalyzeSchema = z.object({
  fileContent: z.string().min(1),
  fileType: DocumentFileTypeEnum,
  fileName: z.string().optional(),
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  hint: z.string().optional(),
});

// ── /api/document-process ─────────────────────────────────────────────────────
// When `structuredRows` is provided, `fileContent` is ignored and the rows are
// sent to the LLM as a JSON payload (better than re-serializing them as text).
export const DocumentProcessSchema = z.object({
  fileContent: z.string().default(""),
  fileType: DocumentFileTypeEnum,
  fileName: z.string().optional(),
  provider: z.string().min(1),
  model: z.string().min(1),
  apiKey: z.string().default(""),
  baseUrl: z.string().optional(),
  systemPrompt: z.string().min(1),
  structuredRows: z.array(z.record(z.string(), z.unknown())).optional(),
  ...LlmTuningFields,
}).refine(
  (v) => v.fileContent.length > 0 || (v.structuredRows && v.structuredRows.length > 0),
  { message: "Either fileContent or structuredRows must be provided", path: ["fileContent"] }
);

// ── /api/runs POST ────────────────────────────────────────────────────────────
export const RunCreateSchema = z.object({
  sessionId: z.string().optional(),
  runType: z.string().default('unknown'),
  provider: z.string().default('openai'),
  model: z.string().default('unknown'),
  temperature: z.number().optional(),
  maxTokens: z.number().int().positive().optional(),
  systemPrompt: z.string().optional(),
  schemaJson: z.string().optional(),
  variablesJson: z.string().optional(),
  inputFile: z.string().default('unnamed'),
  inputRows: z.number().int().min(0).default(0),
  jsonMode: z.boolean().optional(),
  maxConcurrency: z.number().int().positive().optional(),
  config: z.string().optional(),
});

// ── /api/results POST ─────────────────────────────────────────────────────────
export const ResultsBatchSchema = z.object({
  runId: z.string().min(1),
  results: z.array(
    z.object({
      rowIndex: z.number().int(),
      input: z.record(z.string(), z.unknown()),
      output: z.union([z.string(), z.record(z.string(), z.unknown())]),
      status: z.string().default('success'),
      latency: z.number().optional(),
      errorType: z.string().optional(),
      errorMessage: z.string().optional(),
    })
  ),
});
