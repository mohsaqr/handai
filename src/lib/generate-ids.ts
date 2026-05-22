// Continuous ID columns for the Generate tool.
//
// Synthetic data is produced in independent batches (one LLM call each), so an
// ID-like column restarts at 1 every batch (1..25, then 1..25 again — duplicate
// IDs). These helpers detect such columns from the first clean batch — a
// constant text wrapper around a single digit run that increments by a fixed
// step, e.g. "TK2026-001", "G001", or "42" — then overwrite them with a global
// running counter so IDs stay unique and continuous (…025, 026…), preserving the
// model's own prefix and zero-padding.

export interface ParsedId {
  prefix: string;
  num: number;
  suffix: string;
  /** The raw digit run, kept so detection can tell zero-padding apart from
   *  natural digit growth (e.g. "001" vs "1"). */
  digits: string;
}

export interface IdPattern {
  prefix: string;
  suffix: string;
  /** Min digits to print. Equals the fixed pad width for zero-padded ids
   *  ("001" → 3); 1 for unpadded ids so they print naturally (1, 2, … 10, 11). */
  width: number;
  step: number;
  start: number;
}

// Capture the LAST digit run as the counter; the `(\D*)$` anchor forces any
// digits inside the prefix (e.g. the "2026" in "TK2026-001") to stay in the
// prefix, so only the trailing "001" is treated as the incrementing part.
export function parseIdValue(v: unknown): ParsedId | null {
  if (typeof v !== "string" && typeof v !== "number") return null;
  const m = String(v).match(/^(.*?)(\d+)(\D*)$/);
  if (!m) return null;
  return { prefix: m[1], num: parseInt(m[2], 10), suffix: m[3], digits: m[2] };
}

export const ID_NAME_RE = /(^|[_\s-])(id|no|num|number|index|idx|code|ref|key)([_\s-]|$)/i;

const isPadded = (digits: string) => digits.length > 1 && digits.startsWith("0");

// A column qualifies as an ID if every value shares the same prefix/suffix AND
// the numbers climb by a constant positive step. Zero-padded ids must keep a
// fixed digit width ("001","002"…); unpadded ids may grow naturally (9 → 10)
// but must not mix in leading zeros. A pure step-of-1 sequence counts as an id
// even without a telltale name (covers plain "1,2,3…" row numbers); other steps
// require an id-ish column name so we don't grab a coincidentally-ordered data
// column.
export function detectIdPatterns(
  rows: Record<string, unknown>[],
): Record<string, IdPattern> {
  const out: Record<string, IdPattern> = {};
  if (rows.length === 0) return out;
  for (const key of Object.keys(rows[0])) {
    const parsed = rows.map((r) => parseIdValue(r[key]));
    if (parsed.some((p) => p === null)) continue;
    const ps = parsed as ParsedId[];
    const { prefix, suffix } = ps[0];
    if (!ps.every((p) => p.prefix === prefix && p.suffix === suffix)) continue;

    const padded = isPadded(ps[0].digits);
    const width = padded ? ps[0].digits.length : 1;
    if (padded) {
      if (!ps.every((p) => p.digits.length === width)) continue;
    } else if (ps.some((p) => isPadded(p.digits))) {
      continue;
    }

    let step = 1;
    if (ps.length >= 2) {
      step = ps[1].num - ps[0].num;
      if (!ps.every((p, i) => i === 0 || p.num - ps[i - 1].num === step)) continue;
    }
    if (step < 1) continue;
    if (step !== 1 && !ID_NAME_RE.test(key)) continue;
    out[key] = { prefix, suffix, width, step, start: ps[0].num };
  }
  return out;
}

// Overwrite detected ID columns in place with a global counter keyed off each
// row's absolute position, so the sequence continues across batch boundaries.
// Pass `startIndex` to renumber only rows from that index on — the earlier rows
// are already correct (and, on resume, must not be rewritten).
export function applyIdPatterns(
  rows: Record<string, unknown>[],
  patterns: Record<string, IdPattern>,
  startIndex = 0,
): void {
  for (const [key, p] of Object.entries(patterns)) {
    for (let i = startIndex; i < rows.length; i++) {
      rows[i][key] = `${p.prefix}${String(p.start + i * p.step).padStart(p.width, "0")}${p.suffix}`;
    }
  }
}
