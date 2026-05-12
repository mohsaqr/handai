/**
 * Helpers for sending already-structured tabular data (CSV / XLSX / XLS / JSON / RIS)
 * to the LLM as a structured JSON payload instead of as raw text.
 *
 * Used by both browser-direct (`documentExtractDirect` in `llm-browser.ts`)
 * and server (`api/document-extract/route.ts`) paths.
 */

import type { FieldDef } from "@/types";
import { formatExtractionSchemaJson } from "@/lib/prompts";

type Row = Record<string, unknown>;

/** Target JSON payload size per chunk (chars), counting only the rows array.
 * Tuned to keep prompt+response well under typical 128k-token context windows
 * while still batching enough rows that multi-chunk overhead is small. */
export const STRUCTURED_CHUNK_TARGET_CHARS = 6_000;

/** Maximum rows per chunk regardless of size — guards against huge per-row JSON
 * pushing a single chunk above the model's output cap. */
export const STRUCTURED_CHUNK_MAX_ROWS = 200;

/** Union of all column names that appear across the rows, preserving first-seen order. */
export function collectColumns(rows: Row[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) { seen.add(k); out.push(k); }
    }
  }
  return out;
}

/** Split rows into chunks by approximate JSON character size, with a hard row cap. */
export function chunkRows(rows: Row[], targetChars = STRUCTURED_CHUNK_TARGET_CHARS, maxRows = STRUCTURED_CHUNK_MAX_ROWS): Row[][] {
  if (rows.length === 0) return [];
  const chunks: Row[][] = [];
  let current: Row[] = [];
  let currentChars = 0;
  for (const row of rows) {
    const rowChars = JSON.stringify(row).length;
    const wouldOverflow = current.length > 0 && (currentChars + rowChars > targetChars || current.length >= maxRows);
    if (wouldOverflow) {
      chunks.push(current);
      current = [row];
      currentChars = rowChars;
    } else {
      current.push(row);
      currentChars += rowChars;
    }
  }
  if (current.length > 0) chunks.push(current);
  return chunks;
}

/** System prompt for mapping structured rows to user-defined fields. */
export function buildStructuredSystemPrompt(fields: FieldDef[]): string {
  const fieldList = fields
    .map((f) => `- "${f.name}" (${f.type})${f.description ? ": " + f.description : ""}`)
    .join("\n");
  const schema = formatExtractionSchemaJson(fields);
  return `You are a structured-data field mapper. The input is a table: an array of records where each record is one row.

Your job: output ONE mapped record per input record, in the same order, with the requested fields.

FIELDS TO EXTRACT (use these exact JSON keys):
${fieldList}

Each output object must follow this shape:
${schema}

ABSOLUTE RULES:
1. Return one output object per input row, in the same order as the input.
2. Match input columns to fields semantically, not by literal name (e.g. "Airline Name" → "name", "Annual Revenue (Millions)" → "revenue").
3. Coerce types: numbers as JSON numbers (not strings); strings quoted; missing values as null.
4. If no input column maps to a requested field, use null for that field.
5. Your entire response MUST be a single JSON array. First character "[", last character "]".
6. No prose, no markdown, no code fences, no commentary before or after the array.
7. Never drop, summarize, or merge input rows — one input row = one output object.`;
}

/** User prompt body for a single chunk of structured rows. */
export function buildStructuredUserPrompt(opts: {
  fileName: string;
  columns: string[];
  rows: Row[];
  rowOffset: number;
  totalRows: number;
}): string {
  const { fileName, columns, rows, rowOffset, totalRows } = opts;
  const rangeLabel = totalRows > rows.length
    ? `Rows ${rowOffset + 1}–${rowOffset + rows.length} of ${totalRows}`
    : `Rows (${rows.length} total)`;
  return `File: ${fileName}
Columns: ${columns.join(", ")}
${rangeLabel}:
${JSON.stringify(rows, null, 2)}`;
}
