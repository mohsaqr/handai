/**
 * Browser-side LLM functions — mirror of the six /api/* route handlers.
 *
 * Used in static builds (GitHub Pages, no API routes).
 * llm-dispatch.ts routes here when NEXT_PUBLIC_STATIC is set.
 *
 * All utilities used here (getModel, withRetry, cohenKappa, pairwiseAgreement)
 * are pure fetch / pure JS — no Node.js APIs — so they run in the browser as-is.
 */

import { generateText } from "ai";
import { getModel } from "./ai/providers";
import { withRetry } from "./retry";
import { pairwiseJaccard, pairwiseAgreement, interpretKappa } from "./analytics";
import { getPrompt, formatExtractionSchemaJson } from "./prompts";
import type { FieldDef } from "@/types";
import { chunkText, chunkPromptPrefix, CHUNK_CONCURRENCY } from "./chunk-text";
import { getFileExt, isStructuredExt, parseStructuredBuffer } from "./parse-file";
import { buildStructuredSystemPrompt, buildStructuredUserPrompt, chunkRows, collectColumns } from "./structured-extract";
import pLimit from "p-limit";

/** Build generateText options — only includes temperature when explicitly provided (reasoning models reject it). */
function genOpts(model: ReturnType<typeof getModel>, system: string, prompt: string, temperature?: number, maxOutputTokens?: number): Parameters<typeof generateText>[0] {
  const opts: Parameters<typeof generateText>[0] = { model, system, prompt };
  if (temperature !== undefined && temperature !== null) opts.temperature = temperature;
  if (maxOutputTokens !== undefined) opts.maxOutputTokens = maxOutputTokens;
  return opts;
}

// ── Types ──────────────────────────────────────────────────────────────────────

export interface WorkerResult {
  id: string;
  output: string;
  latency: number;
}

export interface AgentNetworkResult {
  roundOutputs: Array<{
    round: number;
    outputs: Array<{ label: string; output: string; latency: number }>;
  }>;
  roundsTaken: number;
  converged: boolean;
  finalConsensus: string | null;
}

export interface ConsensusResult {
  workerResults: WorkerResult[];
  reconcilerOutput: string;
  reconcilerReasoning?: string;
  reconcilerLatency: number;
  consensusType: string;
  kappa: number | null;
  kappaLabel: string;
  agreementMatrix: ReturnType<typeof pairwiseAgreement>;
  qualityScores?: (number | null)[];
  disagreementReason?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function parseRow(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
      else { inQuotes = !inQuotes; }
    } else if (ch === "," && !inQuotes) {
      values.push(current.trim()); current = "";
    } else {
      current += ch;
    }
  }
  values.push(current.trim());
  return values;
}

function parseCsv(text: string, expectedColumns?: string[]): Record<string, string>[] {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length < 2) return [];

  const aiHeaders = parseRow(lines[0]);

  if (expectedColumns && expectedColumns.length > 0) {
    const aiLower = aiHeaders.map((h) => h.toLowerCase());
    const colIndex = new Map<string, number>();
    for (const col of expectedColumns) {
      const idx = aiLower.indexOf(col.toLowerCase());
      if (idx !== -1) colIndex.set(col, idx);
    }

    return lines.slice(1).map((line) => {
      const values = parseRow(line);
      const row: Record<string, string> = {};
      for (const col of expectedColumns) {
        const idx = colIndex.get(col);
        row[col] = idx !== undefined ? (values[idx] ?? "") : "";
      }
      return row;
    });
  }

  return lines.slice(1).map((line) => {
    const values = parseRow(line);
    const row: Record<string, string> = {};
    aiHeaders.forEach((h, idx) => { row[h] = values[idx] ?? ""; });
    return row;
  });
}

function parseJsonLines(text: string, expectedColumns?: string[]): Record<string, string>[] {
  let objects: Record<string, unknown>[];

  const cleaned = text.replace(/^```(?:json(?:l)?)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();

  try {
    const parsed = JSON.parse(cleaned);
    objects = Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    objects = cleaned
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.startsWith("{"))
      .map((l) => {
        try { return JSON.parse(l); } catch { return null; }
      })
      .filter((o): o is Record<string, unknown> => o !== null);
  }

  if (objects.length === 0) return [];

  return objects.map((obj) => {
    const row: Record<string, string> = {};
    if (expectedColumns && expectedColumns.length > 0) {
      const keyMap = new Map<string, string>();
      for (const k of Object.keys(obj)) keyMap.set(k.toLowerCase(), k);
      for (const col of expectedColumns) {
        const actualKey = keyMap.get(col.toLowerCase());
        row[col] = actualKey !== undefined ? String(obj[actualKey] ?? "") : "";
      }
    } else {
      for (const [k, v] of Object.entries(obj)) {
        row[k] = String(v ?? "");
      }
    }
    return row;
  });
}

function extractJson(text: string): Record<string, unknown> | null {
  try { return JSON.parse(text); } catch { /* fall through */ }
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) { try { return JSON.parse(fenceMatch[1]); } catch { /* fall through */ } }
  const braceMatch = text.match(/\{[\s\S]*\}/);
  if (braceMatch) { try { return JSON.parse(braceMatch[0]); } catch { /* fall through */ } }
  return null;
}

// ── processRowDirect — mirrors /api/process-row ────────────────────────────────

export async function processRowDirect(params: {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  systemPrompt: string;
  userContent: string;
  temperature?: number;
  maxTokens?: number;
}): Promise<{ output: string; latency: number }> {
  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  const start = Date.now();
  const genOpts: Parameters<typeof generateText>[0] = {
    model: aiModel,
    system: params.systemPrompt,
    prompt: params.userContent,
    maxOutputTokens: params.maxTokens ?? undefined,
  };
  if (params.temperature !== undefined) {
    genOpts.temperature = params.temperature;
  }
  const { text } = await withRetry(
    () => generateText(genOpts),
    { maxAttempts: 3, baseDelayMs: 100 }
  );
  return { output: text, latency: (Date.now() - start) / 1000 };
}

// ── generateRowDirect — mirrors /api/generate-row ─────────────────────────────

export async function generateRowDirect(params: {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  rowCount: number;
  columns?: Array<{ name: string; type: string; description?: string }>;
  freeformPrompt?: string;
  outputFormat?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
}): Promise<{ rows: Record<string, string>[]; rawCsv: string; count: number; raw?: string }> {
  const isFreetext = params.outputFormat === "freetext" || params.outputFormat === "markdown" || params.outputFormat === "gift";
  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);

  let systemPrompt: string;
  let userPrompt: string;

  if (isFreetext) {
    systemPrompt = params.systemPrompt || getPrompt(params.outputFormat === "markdown" ? "generate.markdown" : params.outputFormat === "gift" ? "generate.gift" : "generate.freetext");
    userPrompt = params.freeformPrompt ?? "Generate realistic content.";
  } else if (params.systemPrompt) {
    systemPrompt = params.systemPrompt;
    if (params.columns && params.columns.length > 0) {
      const colDefs = params.columns
        .map((c) => `${c.name} (${c.type}${c.description ? `: ${c.description}` : ""})`)
        .join(", ");
      const colNames = params.columns.map((c) => c.name).join('", "');
      userPrompt = `Generate ${params.rowCount} rows of realistic data as a JSON array.\nColumns: ${colDefs}\nJSON keys must be: ["${colNames}"]`;
    } else {
      userPrompt = `${params.freeformPrompt ?? "Generate a realistic dataset"}\nGenerate exactly ${params.rowCount} rows as a JSON array.`;
    }
  } else if (params.columns && params.columns.length > 0) {
    systemPrompt = getPrompt("generate.csv_with_cols");
    const colDefs = params.columns
      .map((c) => `${c.name} (${c.type}${c.description ? `: ${c.description}` : ""})`)
      .join(", ");
    const colNames = params.columns.map((c) => c.name).join('", "');
    userPrompt = `Generate ${params.rowCount} rows of realistic data.\nColumns: ${colDefs}\nJSON keys must be: ["${colNames}"]`;
  } else {
    systemPrompt = getPrompt("generate.csv_freeform");
    userPrompt = `${params.freeformPrompt ?? "Generate a realistic dataset"}\nGenerate exactly ${params.rowCount} rows.`;
  }

  const { text } = await withRetry(
    () => generateText(genOpts(aiModel, systemPrompt, userPrompt, params.temperature, params.maxTokens ?? undefined)),
    { maxAttempts: 3, baseDelayMs: 200 }
  );

  if (isFreetext) {
    return { rows: [], rawCsv: text, count: 0, raw: text };
  }

  const cleaned = text.replace(/^```(?:json(?:l)?|csv)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();
  const expectedCols = params.columns?.map((c) => c.name);
  let rows = parseJsonLines(cleaned, expectedCols);
  if (rows.length === 0) {
    rows = parseCsv(cleaned, expectedCols);
  }
  if (rows.length === 0 && cleaned.length > 0) {
    // Last resort: treat the entire output as a single-cell row so data isn't lost
    rows = [{ output: cleaned }];
  }
  return { rows, rawCsv: cleaned, count: rows.length };
}

// ── consensusRowDirect — mirrors /api/consensus-row ───────────────────────────

export async function consensusRowDirect(params: {
  workers: Array<{ provider: string; model: string; apiKey: string; baseUrl?: string; persona?: string }>;
  reconciler: { provider: string; model: string; apiKey: string; baseUrl?: string; persona?: string };
  workerPrompt: string;
  reconcilerPrompt: string;
  userContent: string;
  enableQualityScoring?: boolean;
  enableDisagreementAnalysis?: boolean;
  includeReasoning?: boolean;
  temperature?: number;
  maxTokens?: number;
}): Promise<ConsensusResult> {
  // Default direct-answer-only rules — overridden when the instruction above explicitly asks for reasoning
  const strictSuffix = `\n\nDEFAULT OUTPUT RULES (apply only when the instruction above does NOT explicitly request explanations, reasoning, arguments, justification, or rationale; if it does, follow the instruction and ignore these rules):
- Output ONLY the answer to the instruction. No notes, no explanations, no reasoning, no commentary, no caveats.
- Plain text or CSV only. NEVER use markdown: no **, no ## headings, no bullet points, no code blocks, no backticks.
- Do NOT add headers, labels, introductions, or sign-offs.
- Do NOT prefix with "Answer:", "Result:", "Note:", or any label.
- Do NOT add extra sentences, context, or qualifications.
- If the instruction asks for a single value, return that value and NOTHING else.`;

  // Step 1: Run workers in parallel
  const enforcedWorkerPrompt = params.workerPrompt + strictSuffix;
  const workerPromises = params.workers.map(async (w, i) => {
    const model = getModel(w.provider, w.model, w.apiKey || "local", w.baseUrl);
    const workerSystem = w.persona ? `${w.persona}\n\n${enforcedWorkerPrompt}` : enforcedWorkerPrompt;
    const start = Date.now();
    const { text } = await withRetry(
      () => generateText(genOpts(model, workerSystem, params.userContent, params.temperature, params.maxTokens)),
      { maxAttempts: 3, baseDelayMs: 100 }
    );
    return { id: `worker_${i + 1}`, output: text, latency: (Date.now() - start) / 1000 };
  });

  const workerSettled = await Promise.allSettled(workerPromises);
  const workerResults = workerSettled
    .filter((r): r is PromiseFulfilledResult<WorkerResult> => r.status === "fulfilled")
    .map((r) => r.value);

  if (workerResults.length < 1) {
    const errors = workerSettled
      .filter((r): r is PromiseRejectedResult => r.status === "rejected")
      .map((r) => (r.reason instanceof Error ? r.reason.message : String(r.reason)));
    throw new Error(
      `Not enough workers succeeded (${workerResults.length}/${params.workers.length}). Errors: ${errors.join("; ")}`
    );
  }

  // Step 2: Inter-rater analytics (all workers, set-based)
  const outputs = workerResults.map((r) => r.output.trim());
  const allSame = outputs.every((o) => o === outputs[0]);
  const kappa = pairwiseJaccard(outputs);

  const allTokenized = outputs.map((o) =>
    o.split(/[,\n]+/).map((s) => s.trim()).filter(Boolean)
  );
  const maxAllLen = Math.max(...allTokenized.map((t) => t.length), 1);
  const padded = allTokenized.map((t) =>
    t.concat(new Array(Math.max(0, maxAllLen - t.length)).fill(""))
  );
  const agreementMatrix = pairwiseAgreement(padded);

  // Step 3: Run reconciler — also classify consensus level
  const reconcilerModel = getModel(
    params.reconciler.provider,
    params.reconciler.model,
    params.reconciler.apiKey || "local",
    params.reconciler.baseUrl
  );
  const workersFormatted = workerResults
    .map((r) => `${r.id} response:\n${r.output}`)
    .join("\n\n---\n\n");
  const combinedContent = `Worker Instruction: ${params.workerPrompt}\n\nOriginal Data: ${params.userContent}\n\nWorker Responses:\n${workersFormatted}`;
  const reconcilerPersonaPrefix = params.reconciler.persona ? `${params.reconciler.persona}\n\n` : "";

  const taskContext = `\n\nTHE WORKERS WERE GIVEN THIS INSTRUCTION:
${params.workerPrompt}

GROUNDING RULE (highest priority — read carefully):
- Your output MUST be derived ONLY from the content of the Worker Responses.
- The "Original Data" is provided ONLY as context so you can understand what the workers were given. You MUST NOT use it as a source for your own output.
- Do NOT translate, summarize, classify, rewrite, or otherwise transform the Original Data yourself. That was the workers' job, not yours.
- Do NOT add facts, claims, or content that no worker provided. If a detail is not present in any worker's output, it must NOT appear in yours.
- Your output's language and format MUST match what the worker outputs use. If workers output English text, you output English text. If workers output keywords or codes, you output keywords or codes. Do NOT translate, paraphrase, or transform worker outputs into a different language or form — even if the worker instruction asked for that language/form.
- Your job is to judge or combine the workers' outputs — never to redo the workers' task. You are not allowed to "fix" or "complete" a worker's failure to comply.

TASK COMPLIANCE GUIDANCE:
- Use the worker instruction above to recognize whether each worker actually performed the task. A worker that returned the Original Data verbatim, returned content in the wrong language/format, or refused has NOT complied.
- "Accuracy" means accuracy with respect to the worker instruction applied to the data, NOT similarity to the Original Data itself.
- Prefer outputs from workers that complied with the instruction. If only one worker complied, base your answer on that worker.
- If NO worker complied, you must STILL stay grounded in the worker outputs as they are. Pick or merge from what the workers actually produced — do NOT silently perform the task yourself on the Original Data, and do NOT translate or transform worker outputs to compensate for their failure.`;

  let reconcilerOutput: string;
  let reconcilerLatency: number;
  let consensusType: string;
  let reconcilerReasoning: string | undefined;

  const reconcilerDirectSuffix = `\n\nDEFAULT OUTPUT RULES (apply only when the worker instruction above does NOT explicitly request explanations, reasoning, arguments, justification, or rationale; if it does, mirror that level of detail in your final answer and ignore these rules):
- Output ONLY the final answer. No explanations, no reasoning, no commentary, no justifications.
- Plain text only. No markdown, no headings, no bullet points, no code fences.
- Do NOT explain why you chose this answer — just give the answer directly.`;

  if (allSame) {
    consensusType = "Unanimous";
    const reconcilerStart = Date.now();
    const { text } = await withRetry(
      () => generateText(genOpts(reconcilerModel, reconcilerPersonaPrefix + params.reconcilerPrompt + taskContext + reconcilerDirectSuffix, combinedContent, params.temperature, params.maxTokens)),
      { maxAttempts: 3, baseDelayMs: 100 }
    );
    reconcilerOutput = text;
    reconcilerLatency = (Date.now() - reconcilerStart) / 1000;
  } else {
    const reconcilerSuffix = reconcilerDirectSuffix + `\n\nADDITIONAL TASK — After producing your direct answer, you MUST end your response with a consensus classification on its own line, in this exact format:
[CONSENSUS: <level>]
Where <level> is one of:
- "Unanimous" — all workers conveyed the same meaning (even if worded differently)
- "Strong Agreement" — workers mostly agree with only minor differences in detail or phrasing
- "Partial Agreement" — workers agree on some points but differ on others
- "Divergent" — workers gave substantially different or contradictory responses`;

    const reconcilerStart = Date.now();
    const { text: rawReconciler } = await withRetry(
      () => generateText(genOpts(reconcilerModel, reconcilerPersonaPrefix + params.reconcilerPrompt + taskContext + reconcilerSuffix, combinedContent, params.temperature, params.maxTokens)),
      { maxAttempts: 3, baseDelayMs: 100 }
    );
    reconcilerLatency = (Date.now() - reconcilerStart) / 1000;

    // Parse [CONSENSUS: ...] — tolerant of missing ], quotes, trailing whitespace
    const consensusMatch = rawReconciler.match(/\[CONSENSUS:\s*"?([^"\]\n]+)"?\]?\s*$/im);
    if (consensusMatch) {
      const level = consensusMatch[1].trim();
      const valid = ["Unanimous", "Strong Agreement", "Partial Agreement", "Divergent"];
      consensusType = valid.includes(level) ? level : "Partial Agreement";
      reconcilerOutput = rawReconciler.slice(0, consensusMatch.index).trim();
    } else {
      consensusType = "Partial Agreement";
      reconcilerOutput = rawReconciler.trim();
    }
  }
  // Step 4 (optional): Quality scoring
  let qualityScores: (number | null)[] | undefined;
  if (params.enableQualityScoring) {
    try {
      const { text: qsText } = await withRetry(
        () =>
          generateText(genOpts(reconcilerModel, `You are a quality assessor evaluating worker responses. Rate each worker on a scale of 1-10.

SCORING CRITERIA:
- Task compliance (did the worker actually perform the Worker Instruction on the Original Data? A response that returns the Original Data verbatim, refuses, or is in the wrong language fails the instruction and must score very low — typically 1-3.)
- Accuracy (is the response correct with respect to the Worker Instruction applied to the Original Data — NOT similarity to the Original Data itself?)
- Completeness (does it cover all relevant aspects required by the instruction?)
- Relevance (does it stay focused on the task?)
- Alignment with best answer (how close is it to the chosen reconciler output?)

RULES:
- If all workers gave the same answer, they should all receive the same score.
- A response that matches the reconciler's chosen answer closely should score higher.
- A response that did not perform the requested task is invalid — score it 1-3 regardless of fluency.
- Deduct points for: factual errors, missing key information, off-topic content, unnecessary additions.
- Be consistent: similar quality responses should get similar scores.

Return ONLY valid JSON: {"quality_scores":[N,N,...]} where N is a decimal number between 1.0 and 10.0 (one decimal place, e.g. 6.5, 7.3, 9.0). No other text.`, `Worker Instruction: ${params.workerPrompt}\n\nOriginal Data: ${params.userContent}\n\nWorker Responses:\n${workersFormatted}\n\nReconciler's Chosen Answer:\n${reconcilerOutput}\n\nConsensus Level: ${consensusType}`, params.temperature, params.maxTokens)),
        { maxAttempts: 2, baseDelayMs: 100 }
      );
      const cleaned = qsText.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
      const tryParse = (s: string): { quality_scores: unknown } | null => {
        try { return JSON.parse(s) as { quality_scores: unknown }; } catch { return null; }
      };
      let parsed = tryParse(cleaned);
      if (!parsed) {
        const objMatch = cleaned.match(/\{[\s\S]*\}/);
        if (objMatch) parsed = tryParse(objMatch[0]);
      }
      if (!parsed) {
        const arrMatch = cleaned.match(/\[[\s\S]*?\]/);
        if (arrMatch) parsed = tryParse(`{"quality_scores":${arrMatch[0]}}`);
      }

      if (parsed && Array.isArray(parsed.quality_scores)) {
        const raw = parsed.quality_scores as unknown[];
        const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
        const normalize = (s: string) => s.trim().replace(/\s+/g, " ").toLowerCase();
        const groups = new Map<string, number[]>();
        outputs.forEach((out, i) => {
          const v = raw[i];
          if (!isFiniteNum(v)) return;
          const key = normalize(out);
          const bucket = groups.get(key);
          if (bucket) bucket.push(v);
          else groups.set(key, [v]);
        });
        qualityScores = outputs.map((out, i) => {
          const bucket = groups.get(normalize(out));
          if (bucket && bucket.length > 0) {
            const avg = bucket.reduce((a, b) => a + b, 0) / bucket.length;
            return Math.round(avg * 10) / 10;
          }
          const v = raw[i];
          return isFiniteNum(v) ? v : null;
        });
      } else {
        console.warn("[consensusRowDirect] quality_scores parse failed:", qsText.slice(0, 300));
        qualityScores = outputs.map(() => null);
      }
    } catch (err) {
      console.warn("[consensusRowDirect] quality scoring error:", err instanceof Error ? err.message : String(err));
      qualityScores = outputs.map(() => null);
    }
  }

  // Step 5 (optional): Reconciler reasoning — separate call for reliability
  if (params.includeReasoning && !allSame) {
    try {
      const { text: jrText } = await withRetry(
        () =>
          generateText(genOpts(reconcilerModel, `You are a reconciler explaining your decision. Given the original data, the worker responses, and your chosen best answer, explain in one or two sentences why you chose this answer over the alternatives. Return ONLY the explanation, no labels or prefixes.`, `Worker Instruction: ${params.workerPrompt}\n\nOriginal Data: ${params.userContent}\n\nWorker Responses:\n${workersFormatted}\n\nChosen Answer:\n${reconcilerOutput}`, params.temperature, params.maxTokens)),
        { maxAttempts: 2, baseDelayMs: 100 }
      );
      reconcilerReasoning = jrText.trim() || "Could not generate reasoning";
    } catch {
      reconcilerReasoning = "Could not generate reasoning";
    }
  }

  // Step 6 (optional): Disagreement analysis
  let disagreementReason: string | undefined;
  if (params.enableDisagreementAnalysis && consensusType !== "Unanimous") {
    try {
      const { text: drText } = await withRetry(
        () =>
          generateText(genOpts(reconcilerModel, `You are an expert analyst. In exactly one sentence, explain why the workers disagreed.`, `Worker Instruction: ${params.workerPrompt}\n\nOriginal Data: ${params.userContent}\n\nWorker Responses:\n${workersFormatted}`, params.temperature, params.maxTokens)),
        { maxAttempts: 2, baseDelayMs: 100 }
      );
      disagreementReason = drText.trim() || "Could not analyze disagreement";
    } catch {
      disagreementReason = "Could not analyze disagreement";
    }
  }

  return {
    workerResults,
    reconcilerOutput,
    reconcilerLatency,
    consensusType,
    kappa: isNaN(kappa) ? null : kappa,
    kappaLabel: interpretKappa(kappa),
    agreementMatrix,
    ...(reconcilerReasoning !== undefined ? { reconcilerReasoning } : {}),
    ...(qualityScores !== undefined ? { qualityScores } : {}),
    ...(disagreementReason !== undefined ? { disagreementReason } : {}),
  };
}

// ── automatorRowDirect — mirrors /api/automator-row ───────────────────────────

export async function automatorRowDirect(params: {
  row: Record<string, unknown>;
  steps: Array<{
    name: string;
    task: string;
    input_fields: string[];
    output_fields: Array<{ name: string; type: string; constraints?: string }>;
  }>;
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
}): Promise<{
  output: Record<string, unknown>;
  stepResults: Array<{ step: string; output?: unknown; raw?: string; success: boolean; error?: string }>;
  success: boolean;
}> {
  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  let cumulativeData: Record<string, unknown> = { ...params.row };
  const stepResults: Array<{ step: string; output?: unknown; raw?: string; success: boolean; error?: string }> = [];

  for (const step of params.steps) {
    const inputData: Record<string, unknown> = {};
    if (step.input_fields.length > 0) {
      step.input_fields.forEach((f) => { inputData[f] = cumulativeData[f]; });
    } else {
      Object.assign(inputData, cumulativeData);
    }

    const expectedFields = step.output_fields.map((f) => f.name);
    const schemaHint = step.output_fields
      .map((f) => `"${f.name}": <${f.type}>${f.constraints ? ` (${f.constraints})` : ""}`)
      .join(",\n  ");
    const systemPrompt = `TASK: ${step.task}

Return a JSON object with EXACTLY these keys — no other keys, no renaming, no translation:
{
  ${schemaHint}
}

RULES:
- Use the EXACT field names shown above (in English, as-is) — never translate or rename keys
- The keys in the input data are field identifiers, NOT content — do not translate or transform them
- Apply the task ONLY to the values, not the keys
- Each value must be a plain ${step.output_fields.length === 1 ? step.output_fields[0].type : "string or number"} — never a nested object
- Do not wrap values in objects like {"text": "..."} — return the value directly
- No markdown, no code fences, no explanation — return ONLY the JSON object`;

    const { text } = await withRetry(
      () =>
        generateText(genOpts(aiModel, systemPrompt, `Input Data (keys are field identifiers — do NOT translate them):\n${JSON.stringify(inputData)}`, params.temperature, params.maxTokens)),
      { maxAttempts: 3, baseDelayMs: 100 }
    );

    const parsedOutput = extractJson(text);
    if (parsedOutput) {
      const normalized: Record<string, unknown> = {};
      for (const fieldName of expectedFields) {
        let val = parsedOutput[fieldName];
        if (val === undefined) {
          const lowerField = fieldName.toLowerCase();
          const matchKey = Object.keys(parsedOutput).find(
            (k) => k.toLowerCase() === lowerField || k.toLowerCase().replace(/[_\s]/g, "") === lowerField.replace(/[_\s]/g, "")
          );
          if (matchKey) val = parsedOutput[matchKey];
        }
        if (val !== null && val !== undefined && typeof val === "object" && !Array.isArray(val)) {
          const entries = Object.entries(val as Record<string, unknown>);
          val = entries.length === 1 ? entries[0][1] : JSON.stringify(val);
        }
        if (Array.isArray(val)) {
          val = val.map((item) => typeof item === "object" ? JSON.stringify(item) : String(item)).join(", ");
        }
        if (val !== undefined) normalized[fieldName] = val;
      }
      cumulativeData = { ...cumulativeData, ...normalized };
      stepResults.push({ step: step.name, output: normalized, success: true });
    } else {
      stepResults.push({ step: step.name, raw: text, success: false, error: "Failed to parse JSON" });
    }
  }

  return { output: cumulativeData, stepResults, success: true };
}

// ── documentExtractDirect — mirrors /api/document-extract ─────────────────────

/** Parse a JSON array out of an LLM response, tolerating code fences and prose wrappers. */
function parseJsonArrayLoose(src: string): Record<string, unknown>[] {
  const cleaned = src.replace(/^```(?:json|csv)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();
  try {
    const parsed = JSON.parse(cleaned);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    const arrMatch = cleaned.match(/\[[\s\S]*\]/);
    if (arrMatch) {
      try { const p = JSON.parse(arrMatch[0]); return Array.isArray(p) ? p : [p]; } catch { /* fall through */ }
    }
    const objMatch = cleaned.match(/\{[\s\S]*\}/);
    if (objMatch) {
      try { return [JSON.parse(objMatch[0])]; } catch { /* fall through */ }
    }
    return [];
  }
}

/** Map LLM-returned keys to defined field names (exact match → fuzzy match)
 * and fill any still-missing fields with "". Throws if every normalized record
 * is entirely empty. */
function normalizeFieldRecords(
  records: Record<string, unknown>[],
  fields: FieldDef[],
  previewSource: string,
): Record<string, unknown>[] {
  const fieldNames = fields.map((f) => f.name);
  const fieldNamesLower = fieldNames.map((n) => n.toLowerCase().replace(/[\s_-]+/g, ""));
  const normalized = records.map((r) => {
    const out: Record<string, unknown> = {};
    for (const f of fieldNames) {
      if (f in r) out[f] = r[f];
    }
    for (const [key, value] of Object.entries(r)) {
      if (fieldNames.includes(key)) continue;
      const keyNorm = key.toLowerCase().replace(/[\s_-]+/g, "");
      for (let i = 0; i < fieldNamesLower.length; i++) {
        if (out[fieldNames[i]] !== undefined) continue;
        if (keyNorm === fieldNamesLower[i] || keyNorm.endsWith(fieldNamesLower[i]) || keyNorm.includes(fieldNamesLower[i])) {
          out[fieldNames[i]] = value;
          break;
        }
      }
    }
    for (const f of fieldNames) {
      if (out[f] === undefined) out[f] = "";
    }
    return out;
  });
  const allEmpty = normalized.every((r) => fieldNames.every((f) => r[f] === "" || r[f] === null || r[f] === undefined));
  if (allEmpty) {
    const preview = previewSource.slice(0, 200).replace(/\s+/g, " ").trim();
    throw new Error(`Model returned no usable field values. The document may lack extractable text (scanned PDF?) or the field definitions don't match its content. Preview: "${preview}${previewSource.length > 200 ? "…" : ""}"`);
  }
  return normalized;
}

/** Send already-parsed structured rows (CSV/XLSX/JSON/RIS) to the LLM as a
 * JSON payload (file title + columns + rows). When no fields are defined, the
 * rows are returned directly — no LLM call. */
async function extractStructured(params: {
  file: File;
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  fields?: FieldDef[];
  maxTokens?: number;
  rows: Record<string, unknown>[];
}): Promise<{
  records: Record<string, unknown>[];
  fileName: string;
  charCount: number;
  truncated: boolean;
  count: number;
  chunks: number;
  failedChunks: number;
}> {
  const { rows, fields } = params;
  const columns = collectColumns(rows);

  // No fields defined → the input columns are already the answer; skip the LLM.
  if (!fields || fields.length === 0) {
    return {
      records: rows,
      fileName: params.file.name,
      charCount: 0,
      truncated: false,
      count: rows.length,
      chunks: 1,
      failedChunks: 0,
    };
  }

  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  const outputOpts: Record<string, unknown> = {};
  if (params.maxTokens) outputOpts.maxOutputTokens = params.maxTokens;

  const systemPrompt = buildStructuredSystemPrompt(fields);
  const rowChunks = chunkRows(rows);

  const runChunk = async (chunkRowsArr: Record<string, unknown>[], rowOffset: number): Promise<Record<string, unknown>[]> => {
    const userPrompt = buildStructuredUserPrompt({
      fileName: params.file.name,
      columns,
      rows: chunkRowsArr,
      rowOffset,
      totalRows: rows.length,
    });
    const { text } = await withRetry(
      () => generateText({ ...genOpts(aiModel, systemPrompt, userPrompt), ...outputOpts }),
      { maxAttempts: 3, baseDelayMs: 200 }
    );
    let recs = parseJsonArrayLoose(text);
    if (recs.length === 0) {
      // Reformat-retry: ask model to coerce its previous output into a JSON array.
      const fieldList = fields
        .map((f) => `- "${f.name}" (${f.type})${f.description ? ": " + f.description : ""}`)
        .join("\n");
      const reformatPrompt = `You are a JSON reformatter. Return ONLY a JSON array of records with these fields:\n${fieldList}\n\nFirst char "[", last char "]". No prose. Use null for missing values.`;
      const { text: reformatted } = await withRetry(
        () => generateText({ ...genOpts(aiModel, reformatPrompt, text), ...outputOpts }),
        { maxAttempts: 2, baseDelayMs: 200 }
      );
      recs = parseJsonArrayLoose(reformatted);
    }
    return recs;
  };

  let records: Record<string, unknown>[] = [];
  let failedChunks = 0;

  if (rowChunks.length === 1) {
    records = await runChunk(rowChunks[0], 0);
  } else {
    const limit = pLimit(CHUNK_CONCURRENCY);
    const offsets: number[] = [];
    let acc = 0;
    for (const c of rowChunks) { offsets.push(acc); acc += c.length; }
    const results = await Promise.allSettled(
      rowChunks.map((c, i) => limit(() => runChunk(c, offsets[i])))
    );
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      if (r.status === "fulfilled") records.push(...r.value);
      else {
        failedChunks++;
        console.error(`documentExtractDirect (structured): chunk ${i + 1}/${rowChunks.length} failed:`, r.reason);
      }
    }
  }

  if (records.length === 0) {
    throw new Error(failedChunks > 0
      ? `All ${rowChunks.length} row batches failed during extraction.`
      : "Model returned unparseable output. Try a stronger model.");
  }

  // Preview source for the "no usable field values" error — first row as JSON
  // is more useful here than a flat text dump since the input was structured.
  const preview = JSON.stringify(rows[0] ?? {});
  records = normalizeFieldRecords(records, fields, preview);

  return {
    records,
    fileName: params.file.name,
    charCount: 0,
    truncated: false,
    count: records.length,
    chunks: rowChunks.length,
    failedChunks,
  };
}

const DEFAULT_EXTRACT_PROMPT = `You are a document data extraction engine. Extract all structured records from the document as a CSV table.

OUTPUT RULES:
1. Output ONLY raw CSV. Nothing else.
2. Row 1: CSV header (design appropriate column names based on the document content).
3. Rows 2+: one extracted record per row, values matching the header columns.
4. Wrap fields containing commas or line breaks in double quotes.

STRICTLY FORBIDDEN: markdown, code blocks, JSON, explanations, or prose.`;

export async function documentExtractDirect(params: {
  file: File;
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  systemPrompt?: string;
  fields?: FieldDef[];
  maxTokens?: number;
}): Promise<{
  records: Record<string, unknown>[];
  fileName: string;
  charCount: number;
  truncated: boolean;
  count: number;
  chunks: number;
  failedChunks: number;
}> {
  // ── Structured-file fast path ──────────────────────────────────────────────
  // For CSV/XLSX/XLS/JSON/RIS we already know the column→value mapping; sending
  // the file to the LLM as raw text would force it to re-parse what we already
  // have. Instead, parse client-side and send the LLM a JSON array of rows.
  const ext = getFileExt(params.file.name);
  if (isStructuredExt(ext)) {
    const buf = await params.file.arrayBuffer();
    const parsedRows = parseStructuredBuffer(buf, ext);
    if (parsedRows && parsedRows.length > 0) {
      return await extractStructured({ ...params, rows: parsedRows });
    }
    // parseStructuredBuffer returned null/[] — fall through to text-based path
  }

  const { extractTextBrowser } = await import("./document-browser");
  const { text: rawText, truncated, charCount } = await extractTextBrowser(params.file);

  if (!rawText.trim()) {
    throw new Error("Document appears to be empty or unreadable");
  }

  // Build effective prompt: fields schema takes priority over custom systemPrompt
  let effectivePrompt: string;
  if (params.fields && params.fields.length > 0) {
    const schema = formatExtractionSchemaJson(params.fields);
    const fieldList = params.fields
      .map((f) => `- "${f.name}" (${f.type})${f.description ? ": " + f.description : ""}`)
      .join("\n");
    effectivePrompt = `You are a data extraction engine. Your ONLY job is to output a JSON array of records.

The document may be a table, a narrative report, a summary, a prose description, or any other format. Extract whatever data matches the requested fields from ANYWHERE in the text — tables, paragraphs, sentences, bullet points, captions, headings, etc. If the document describes a single subject in prose, return one record. If it describes many subjects, return one record per subject.

FIELDS TO EXTRACT (use these exact JSON keys):
${fieldList}

Each object must follow this shape:
${schema}

ABSOLUTE RULES:
1. Your entire response MUST be a single JSON array. The first character must be "[" and the last character must be "]".
2. No prose. No markdown. No code fences. No headings. No explanations before or after the array.
3. Do NOT write a summary of the document — extract the actual field values.
4. If a field value is not present in the document, use null (not an empty string, not "N/A").
5. If the document contains NO relevant data at all, return exactly: []
6. Always wrap records in an array, even when there is only one: [{ ... }]
7. Extract EVERY matching record — there may be dozens or hundreds. Do NOT stop after a sample or subset.
8. Completeness is critical. A long document must produce a long output. Never truncate, summarize, or omit records.`;
  } else {
    effectivePrompt = params.systemPrompt ?? DEFAULT_EXTRACT_PROMPT;
  }

  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  const outputOpts: Record<string, unknown> = {};
  if (params.maxTokens) outputOpts.maxOutputTokens = params.maxTokens;

  const tryJson = (src: string): Record<string, unknown>[] => {
    try {
      const parsed = JSON.parse(src);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      const jsonMatch = src.match(/\[[\s\S]*\]/);
      const objMatch = !jsonMatch ? src.match(/\{[\s\S]*\}/) : null;
      if (jsonMatch) {
        try { const p = JSON.parse(jsonMatch[0]); return Array.isArray(p) ? p : [p]; } catch { /* fall through */ }
      } else if (objMatch) {
        try { return [JSON.parse(objMatch[0])]; } catch { /* fall through */ }
      }
      return [];
    }
  };

  // ── Helper: extract records from a single chunk of text ────────────
  const extractChunk = async (chunk: string, chunkIndex: number, chunkTotal: number): Promise<Record<string, unknown>[]> => {
    const prefix = chunkPromptPrefix(chunkIndex, chunkTotal, "extract");
    const { text } = await withRetry(
      () => generateText({ ...genOpts(aiModel, effectivePrompt, `${prefix}Document: ${params.file.name}\n\n${chunk}`), ...outputOpts }),
      { maxAttempts: 3, baseDelayMs: 200 }
    );

    const cleaned = text.replace(/^```(?:json|csv)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();

    if (params.fields && params.fields.length > 0) {
      let recs = tryJson(cleaned);
      if (recs.length === 0) recs = parseCsv(text);
      if (recs.length === 0) {
        // Reformat-retry
        const fieldList = params.fields
          .map((f) => `- "${f.name}" (${f.type})${f.description ? ": " + f.description : ""}`)
          .join("\n");
        const reformatPrompt = `You are a JSON reformatter. Extract the requested fields and return ONLY a JSON array of records.\n\nREQUIRED FIELDS:\n${fieldList}\n\nRULES:\n1. Output MUST start with "[" and end with "]". Nothing else.\n2. No prose, no markdown, no explanations.\n3. Use null for missing values.`;
        const { text: reformatted } = await withRetry(
          () => generateText({ ...genOpts(aiModel, reformatPrompt, text), ...outputOpts }),
          { maxAttempts: 2, baseDelayMs: 200 }
        );
        recs = tryJson(reformatted.replace(/^```(?:json|csv)?\s*/im, "").replace(/\s*```\s*$/m, "").trim());
      }
      return recs;
    } else {
      let recs: Record<string, unknown>[] = parseCsv(text);
      if (recs.length === 0) recs = tryJson(cleaned);
      if (recs.length === 0) {
        const strippedLines = cleaned.split(/\r?\n/).filter((l: string) =>
          l.trim() && !l.startsWith("Here") && !l.startsWith("The ") && !l.startsWith("Below")
        );
        const retryRecords = parseCsv(strippedLines.join("\n"));
        recs = retryRecords.length > 0 ? retryRecords : [{ extracted_text: text }];
      }
      return recs;
    }
  };

  const chunks = chunkText(rawText);
  let records: Record<string, unknown>[];
  let failedChunks = 0;

  if (chunks.length === 1) {
    records = await extractChunk(rawText, 0, 1);
  } else {
    const limit = pLimit(CHUNK_CONCURRENCY);
    const results = await Promise.allSettled(
      chunks.map((c) => limit(() => extractChunk(c.text, c.index, c.total)))
    );
    records = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      if (r.status === "fulfilled") {
        records.push(...r.value);
      } else {
        failedChunks++;
        console.error(`documentExtractDirect: chunk ${i + 1}/${chunks.length} failed:`, r.reason);
      }
    }
  }

  if (records.length === 0) {
    throw new Error(failedChunks > 0
      ? `All ${chunks.length} sections failed during extraction.`
      : "Model returned unparseable output. Try a stronger model.");
  }

  // Merge single-key objects into one record only when they're true FRAGMENTS
  // of a single record (each contributes a distinct field). Must NOT collapse
  // legitimate single-column multi-record output (e.g. [{name:"Alice"},{name:"Bob"}])
  // where every record has the same key — that's the answer the user asked for.
  if (records.length > 1 && records.every((r: Record<string, unknown>) => Object.keys(r).length === 1)) {
    const allKeys = new Set(records.flatMap((r) => Object.keys(r)));
    if (allKeys.size === records.length) {
      const merged: Record<string, unknown> = {};
      for (const r of records) Object.assign(merged, r);
      records = [merged];
    }
  }

  // Map LLM-returned keys to defined field names and fill missing with ""
  if (params.fields && params.fields.length > 0) {
    records = normalizeFieldRecords(records, params.fields, rawText);
  }

  return { records, fileName: params.file.name, charCount, truncated, count: records.length, chunks: chunks.length, failedChunks };
}

// ── documentAnalyzeDirect — mirrors /api/document-analyze ─────────────────────

export async function documentAnalyzeDirect(params: {
  file: File;
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  hint?: string;
}): Promise<{ fields: FieldDef[] }> {
  // The caller (suggestFields) already shares a fixed character budget across
  // files via distributeBudget, so we forward the prepared combined text as-is.
  const { extractTextBrowser } = await import("./document-browser");
  const { text: rawText } = await extractTextBrowser(params.file);

  if (!rawText.trim()) return { fields: [] };

  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);

  const promptParts = [`Document: ${params.file.name}`];
  if (params.hint) promptParts.push(`\nExtraction goal: ${params.hint}`);
  promptParts.push(`\n\n${rawText}`);

  const { text } = await withRetry(
    () =>
      generateText(genOpts(aiModel, getPrompt("document.analysis"), promptParts.join(""), undefined, 1024)),
    { maxAttempts: 2, baseDelayMs: 200 }
  );

  let fields: FieldDef[] = [];
  try {
    const cleaned = text.replace(/^```(?:json)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();
    const parsed = JSON.parse(cleaned);
    fields = Array.isArray(parsed) ? (parsed as FieldDef[]) : [];
  } catch {
    // Graceful degradation
  }

  return { fields };
}

// ── documentProcessDirect — mirrors /api/document-process ───────────────────

/** Single-call structured-data processing for tabular files (CSV/XLSX/JSON/RIS).
 * Sends the whole table as a JSON payload in one LLM call — no row chunking,
 * because the user's prompt typically requests a single coherent output
 * (CSV with one header, one summary, etc.) and chunked outputs would
 * fragment that. */
async function processStructured(params: {
  file: File;
  rows: Record<string, unknown>[];
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  systemPrompt: string;
  maxTokens?: number;
}): Promise<{
  text: string; fileName: string; charCount: number; truncated: boolean; chunks: number; failedChunks: number;
}> {
  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  const outputOpts: Record<string, unknown> = {};
  if (params.maxTokens) outputOpts.maxOutputTokens = params.maxTokens;

  const columns = collectColumns(params.rows);
  const body = buildStructuredUserPrompt({
    fileName: params.file.name,
    columns,
    rows: params.rows,
    rowOffset: 0,
    totalRows: params.rows.length,
  });
  const userPrompt = `The input is tabular data — columns and rows below are represented as JSON. Apply the system instructions to this data.\n\n${body}`;

  const { text } = await withRetry(
    () => generateText({ ...genOpts(aiModel, params.systemPrompt, userPrompt), ...outputOpts }),
    { maxAttempts: 3, baseDelayMs: 200 }
  );

  if (!text.trim()) {
    throw new Error("Processing produced no output.");
  }

  return {
    text,
    fileName: params.file.name,
    charCount: userPrompt.length,
    truncated: false,
    chunks: 1,
    failedChunks: 0,
  };
}

export async function documentProcessDirect(params: {
  file: File;
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  systemPrompt: string;
  maxTokens?: number;
}): Promise<{
  text: string;
  fileName: string;
  charCount: number;
  truncated: boolean;
  chunks: number;
  failedChunks: number;
}> {
  // Structured-file fast path: parse rows client-side and send the LLM a JSON
  // payload instead of a stringified CSV.
  const ext = getFileExt(params.file.name);
  if (isStructuredExt(ext)) {
    const buf = await params.file.arrayBuffer();
    const parsedRows = parseStructuredBuffer(buf, ext);
    if (parsedRows && parsedRows.length > 0) {
      return await processStructured({ ...params, rows: parsedRows });
    }
    // parseStructuredBuffer returned null/[] — fall through to text-based path
  }

  const { extractTextBrowser } = await import("./document-browser");
  const { text: rawText, truncated, charCount } = await extractTextBrowser(params.file);

  if (!rawText.trim()) {
    throw new Error("Document appears to be empty or unreadable");
  }

  const aiModel = getModel(params.provider, params.model, params.apiKey, params.baseUrl);
  const outputOpts: Record<string, unknown> = {};
  if (params.maxTokens) outputOpts.maxOutputTokens = params.maxTokens;

  const chunks = chunkText(rawText);

  const processChunk = async (chunk: string, chunkIndex: number, chunkTotal: number): Promise<string> => {
    const prefix = chunkPromptPrefix(chunkIndex, chunkTotal, "process");
    const { text } = await withRetry(
      () => generateText({ ...genOpts(aiModel, params.systemPrompt, `${prefix}Document: ${params.file.name}\n\n${chunk}`), ...outputOpts }),
      { maxAttempts: 3, baseDelayMs: 200 }
    );
    return text;
  };

  let text: string;
  let failedChunks = 0;

  if (chunks.length === 1) {
    text = await processChunk(rawText, 0, 1);
  } else {
    const limit = pLimit(CHUNK_CONCURRENCY);
    const results = await Promise.allSettled(
      chunks.map((c) => limit(() => processChunk(c.text, c.index, c.total)))
    );
    const parts: string[] = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      if (r.status === "fulfilled" && r.value.trim()) {
        parts.push(r.value);
      } else if (r.status === "rejected") {
        failedChunks++;
        console.error(`documentProcessDirect: chunk ${i + 1}/${chunks.length} failed:`, r.reason);
        parts.push(`[Section ${i + 1} of ${chunks.length}: processing failed]`);
      }
    }
    text = parts.join("\n\n---\n\n");
  }

  if (!text.trim()) {
    throw new Error("Processing produced no output for any section.");
  }

  return { text, fileName: params.file.name, charCount, truncated, chunks: chunks.length, failedChunks };
}

const COMMUNICATION_STYLE_PROMPTS: Record<string, string> = {
  collaborative: "Review the other agents' outputs below. Incorporate their strongest insights into your answer. Build on areas of agreement, resolve minor differences by merging perspectives, and produce an improved unified response.",
  adversarial: "Review the other agents' outputs below. Critically challenge their claims — identify weaknesses, logical gaps, unsupported assumptions, or factual errors. Defend your own position where you believe you are correct. If another agent makes a valid point that contradicts yours, acknowledge it but explain why your overall conclusion still holds or revise only what is genuinely wrong.",
  deliberative: "Review the other agents' outputs below. Systematically evaluate each agent's position: list the strengths and weaknesses of each perspective. Then produce your revised answer by weighing the evidence objectively. Explain briefly which points you adopted and which you rejected and why.",
  socratic: "Review the other agents' outputs below. For each agent's position, identify the underlying assumptions and question whether they hold. Ask probing questions about their reasoning. Then produce your revised answer that addresses these deeper questions and strengthens your reasoning.",
};

// ── agentNetworkRowDirect — mirrors /api/agent-network-row ───────────────────

export async function agentNetworkRowDirect(params: {
  agents: Array<{
    label: string;
    role: string;
    provider: string;
    model: string;
    apiKey: string;
    baseUrl?: string;
  }>;
  userContent: string;
  maxRounds: number;
  communicationStyle?: string;
  convergenceMode?: "fixed" | "adaptive";
  convergenceThreshold?: number;
  temperature?: number;
  maxTokens?: number;
}): Promise<AgentNetworkResult> {
  if (params.agents.length < 2) throw new Error("Need at least 2 agents");

  const roundOutputs: AgentNetworkResult["roundOutputs"] = [];
  const latestOutputs: Record<string, string> = {};

  let converged = false;
  let roundsTaken = 0;
  const adaptive = params.convergenceMode === "adaptive";
  const threshold = params.convergenceThreshold ?? 15;
  // Adaptive runs up to 5 rounds max (hard cap), but uses delta scoring to stop early
  const hardMaxRounds = adaptive ? 5 : params.maxRounds;

  for (let round = 1; round <= hardMaxRounds; round++) {
    roundsTaken = round;
    const previousOutputs = { ...latestOutputs };

    const promises = params.agents.map(async (agent) => {
      const model = getModel(agent.provider, agent.model, agent.apiKey, agent.baseUrl);

      let agentContent = params.userContent;

      if (round > 1) {
        const othersSection = Object.entries(previousOutputs)
          .filter(([label]) => label !== agent.label)
          .map(([label, output]) => `[${label}]:\n${output}`)
          .join("\n\n");
        const styleInstruction = params.communicationStyle
          ? (COMMUNICATION_STYLE_PROMPTS[params.communicationStyle] || "Refine your answer based on the above.")
          : "Refine your answer based on the above.";
        agentContent += `\n\n--- Other agents' outputs from round ${round - 1} ---\n\n${othersSection}\n\n--- ${styleInstruction} ---`;
      }

      const start = Date.now();
      const { text } = await withRetry(
        () => generateText(genOpts(model, agent.role, agentContent, params.temperature, params.maxTokens)),
        { maxAttempts: 3, baseDelayMs: 100 }
      );
      return { label: agent.label, output: text.trim(), latency: (Date.now() - start) / 1000 };
    });

    const settled = await Promise.allSettled(promises);
    const results = settled
      .filter((r): r is PromiseFulfilledResult<{ label: string; output: string; latency: number }> => r.status === "fulfilled")
      .map((r) => r.value);

    if (results.length === 0) {
      const errors = settled
        .filter((r): r is PromiseRejectedResult => r.status === "rejected")
        .map((r) => (r.reason instanceof Error ? r.reason.message : String(r.reason)));
      throw new Error(`All agents failed in round ${round}: ${errors.join("; ")}`);
    }

    for (const r of results) {
      latestOutputs[r.label] = r.output;
    }

    roundOutputs.push({ round, outputs: results });

    // Strict convergence: all outputs identical to previous round
    if (round > 1) {
      converged = results.every((r) => previousOutputs[r.label] === r.output);
      if (converged) break;
    }

    // Adaptive convergence: judge model scores semantic delta and decides if stable
    if (adaptive && round > 1) {
      try {
        const judgeAgent = params.agents[0];
        const judgeModel = getModel(judgeAgent.provider, judgeAgent.model, judgeAgent.apiKey, judgeAgent.baseUrl);
        const comparison = params.agents.map((a) => {
          const prev = previousOutputs[a.label] ?? "";
          const curr = latestOutputs[a.label] ?? "";
          return `[${a.label}]\nROUND ${round - 1}:\n${prev}\n\nROUND ${round}:\n${curr}`;
        }).join("\n\n---\n\n");

        const { text: judgeText } = await withRetry(
          () => generateText(genOpts(
            judgeModel,
            `You are a convergence detector for a multi-agent deliberation. Compare each agent's output between two consecutive rounds. Score how much agents have changed their positions overall.

Reply with ONLY a JSON object: {"delta": <number 0-100>, "stable": <boolean>}
- delta: 0 = essentially identical (semantic), 100 = completely different positions
- stable: true if agents have converged to stable positions (small revisions only)`,
            comparison,
            0,
            500
          )),
          { maxAttempts: 2, baseDelayMs: 100 }
        );

        let delta = 100;
        try {
          let cleaned = judgeText.trim();
          if (cleaned.startsWith("```")) cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
          const parsed = JSON.parse(cleaned) as { delta: number; stable?: boolean };
          delta = typeof parsed.delta === "number" ? parsed.delta : 100;
        } catch { /* parse failed — keep going */ }

        if (delta <= threshold) {
          converged = true;
          break;
        }
      } catch {
        // Judge failed — continue without adaptive stopping
      }
    }
  }

  const allOutputs = Object.values(latestOutputs);
  const allSame = allOutputs.every((o) => o === allOutputs[0]);
  const finalConsensus = allSame ? allOutputs[0] : null;

  return { roundOutputs, roundsTaken, converged, finalConsensus };
}
