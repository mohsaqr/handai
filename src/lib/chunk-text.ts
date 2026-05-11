/**
 * Smart document chunking for large text extraction.
 *
 * Splits long documents at paragraph boundaries so each chunk produces
 * a manageable number of records for any LLM output-token ceiling.
 * No overlap between chunks — clean splits to avoid duplicate records.
 *
 * Pure logic — no Node.js or browser APIs — usable in both contexts.
 */

/** Documents shorter than this (in characters) are processed in a single call. */
export const CHUNK_THRESHOLD = 10_000;

/** Target size per chunk. */
export const CHUNK_TARGET = 8_000;

/** Max concurrent chunk LLM calls per document to avoid rate-limit storms. */
export const CHUNK_CONCURRENCY = 3;

/**
 * File size (bytes) above which the "Multi-section" upload hint shows.
 * Tuned so typical single-page PDFs/DOCX stay under the threshold; only
 * genuinely large documents that are likely to chunk get flagged.
 */
export const LARGE_FILE_BYTES = 100_000;

export interface TextChunk {
  text: string;
  index: number;
  total: number;
}

/**
 * Split `text` into chunks at paragraph boundaries.
 *
 * Returns a single-element array when the text is below CHUNK_THRESHOLD.
 * Falls back to single-newline splits when no paragraph breaks exist.
 */
export function chunkText(text: string): TextChunk[] {
  if (text.length <= CHUNK_THRESHOLD) {
    return [{ text, index: 0, total: 1 }];
  }

  // Try paragraph boundaries first (double newline)
  let paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim());

  // Fallback: single-newline splits for monolithic text (OCR, HTML-to-text)
  if (paragraphs.length <= 1) {
    paragraphs = text.split(/\n/).filter((p) => p.trim());
  }

  const chunks: string[] = [];
  let current = "";

  for (const para of paragraphs) {
    if (current.length > 0 && current.length + para.length + 2 > CHUNK_TARGET) {
      chunks.push(current);
      current = para;
    } else {
      current += (current ? "\n\n" : "") + para;
    }
  }
  if (current.trim()) {
    chunks.push(current);
  }

  // Last resort: no split points at all — return as single chunk
  if (chunks.length === 0) {
    chunks.push(text);
  }

  return chunks.map((c, i) => ({ text: c, index: i, total: chunks.length }));
}

/**
 * Positional preamble prepended to the user prompt for multi-chunk documents.
 * Returns empty string when total <= 1.
 */
export function chunkPromptPrefix(index: number, total: number, mode: "extract" | "process" = "extract"): string {
  if (total <= 1) return "";
  const section = `[SECTION ${index + 1} OF ${total}]\n`;
  if (mode === "extract") {
    return section + "Extract records ONLY from this section. Do not infer or repeat records from other sections.\n\n";
  }
  return section + "Process only this section. Do not repeat content from other sections.\n\n";
}

/**
 * Whether a file is likely large enough to trigger chunking.
 * Rough heuristic from file size — the actual decision uses character count.
 */
export function isLikelyChunked(fileSize: number): boolean {
  return fileSize > LARGE_FILE_BYTES;
}

/**
 * Distribute a fixed character `budget` across N sources of varying capacity.
 *
 * Each source initially gets a fair share (`budget / N`). Sources that can't
 * use their full share (because their text is shorter) donate the leftover
 * back into the pool, which is then redistributed evenly among sources that
 * still have capacity. Repeats until the budget is exhausted or no source
 * can accept more.
 *
 * Returns an array of allocations the same length as `capacities`, summing
 * to at most `budget` (less only when the total available text is smaller).
 *
 * @example
 * distributeBudget([200, 5000, 5000], 3000) // → [200, 1400, 1400]
 * distributeBudget([10000, 10000], 3000)    // → [1500, 1500]
 */
export function distributeBudget(capacities: number[], budget: number): number[] {
  const allocations = new Array<number>(capacities.length).fill(0);
  if (budget <= 0 || capacities.length === 0) return allocations;

  // Track each source's remaining capacity.
  const remaining = capacities.map((c) => Math.max(0, Math.floor(c)));
  let pool = budget;

  while (pool > 0) {
    const active: number[] = [];
    for (let i = 0; i < remaining.length; i++) if (remaining[i] > 0) active.push(i);
    if (active.length === 0) break;

    const share = Math.floor(pool / active.length);
    if (share === 0) {
      // Less than 1 char per active source — hand out remaining chars one by one.
      const give = Math.min(pool, active.length);
      for (let k = 0; k < give; k++) {
        allocations[active[k]] += 1;
        remaining[active[k]] -= 1;
      }
      pool -= give;
      break;
    }

    let consumed = 0;
    for (const i of active) {
      const take = Math.min(share, remaining[i]);
      allocations[i] += take;
      remaining[i] -= take;
      consumed += take;
    }
    pool -= consumed;
    // Loop continues with the unspent share + any leftovers from already-full sources.
  }

  return allocations;
}
