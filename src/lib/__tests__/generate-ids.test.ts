import { describe, it, expect } from "vitest";
import {
  parseIdValue,
  detectIdPatterns,
  applyIdPatterns,
  type IdPattern,
} from "@/lib/generate-ids";

describe("parseIdValue", () => {
  it("splits a prefixed, zero-padded id into prefix + number + digits", () => {
    expect(parseIdValue("G001")).toEqual({ prefix: "G", num: 1, suffix: "", digits: "001" });
  });

  it("keeps digits that belong to the prefix out of the counter", () => {
    // The trailing run is the counter; "2026" stays in the prefix.
    expect(parseIdValue("TK2026-001")).toEqual({
      prefix: "TK2026-",
      num: 1,
      suffix: "",
      digits: "001",
    });
  });

  it("handles a plain integer", () => {
    expect(parseIdValue("42")).toEqual({ prefix: "", num: 42, suffix: "", digits: "42" });
    expect(parseIdValue(42)).toEqual({ prefix: "", num: 42, suffix: "", digits: "42" });
  });

  it("returns null when there is no digit run", () => {
    expect(parseIdValue("abc")).toBeNull();
    expect(parseIdValue(null)).toBeNull();
    expect(parseIdValue("")).toBeNull();
  });
});

describe("detectIdPatterns", () => {
  it("detects a prefixed sequential id column", () => {
    const rows = [{ id: "TK2026-001" }, { id: "TK2026-002" }, { id: "TK2026-003" }];
    expect(detectIdPatterns(rows)).toEqual({
      id: { prefix: "TK2026-", suffix: "", width: 3, step: 1, start: 1 },
    });
  });

  it("detects a plain step-1 sequence without an id-ish name", () => {
    const rows = [{ row: "1" }, { row: "2" }, { row: "3" }];
    expect(detectIdPatterns(rows).row).toEqual({
      prefix: "",
      suffix: "",
      width: 1,
      step: 1,
      start: 1,
    });
  });

  it("detects unpadded integers that cross a digit-count boundary (1..12)", () => {
    const rows = Array.from({ length: 12 }, (_, i) => ({ id: String(i + 1) }));
    expect(detectIdPatterns(rows).id).toEqual({
      prefix: "",
      suffix: "",
      width: 1,
      step: 1,
      start: 1,
    });
  });

  it("requires an id-ish name when the step is not 1", () => {
    const idish = [{ ticket_no: "5" }, { ticket_no: "10" }, { ticket_no: "15" }];
    expect(detectIdPatterns(idish).ticket_no).toMatchObject({ step: 5, start: 5 });

    const dataCol = [{ price: "5" }, { price: "10" }, { price: "15" }];
    expect(detectIdPatterns(dataCol).price).toBeUndefined();
  });

  it("ignores non-monotonic / reset numeric columns (e.g. a 1..3 rating that repeats)", () => {
    const rows = [{ rating: "1" }, { rating: "2" }, { rating: "3" }, { rating: "1" }];
    expect(detectIdPatterns(rows).rating).toBeUndefined();
  });

  it("ignores columns whose padding width is inconsistent", () => {
    const rows = [{ id: "1" }, { id: "02" }, { id: "003" }];
    expect(detectIdPatterns(rows).id).toBeUndefined();
  });

  it("ignores free-text columns and descending sequences", () => {
    const text = [{ name: "Alice" }, { name: "Bob" }];
    expect(detectIdPatterns(text)).toEqual({});
    const desc = [{ id: "3" }, { id: "2" }, { id: "1" }];
    expect(detectIdPatterns(desc).id).toBeUndefined();
  });
});

describe("applyIdPatterns", () => {
  it("renumbers continuously across what would have been separate batches", () => {
    const pattern: Record<string, IdPattern> = {
      id: { prefix: "TK2026-", suffix: "", width: 3, step: 1, start: 1 },
    };
    // Two batches that both restarted at 1, concatenated.
    const rows = [
      ...Array.from({ length: 3 }, (_, i) => ({ id: `TK2026-00${i + 1}`, v: i })),
      ...Array.from({ length: 3 }, (_, i) => ({ id: `TK2026-00${i + 1}`, v: i + 3 })),
    ];
    applyIdPatterns(rows, pattern);
    expect(rows.map((r) => r.id)).toEqual([
      "TK2026-001",
      "TK2026-002",
      "TK2026-003",
      "TK2026-004",
      "TK2026-005",
      "TK2026-006",
    ]);
    // Non-id columns are untouched.
    expect(rows.map((r) => r.v)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("renumbers only from startIndex, leaving earlier rows untouched", () => {
    const pattern: Record<string, IdPattern> = {
      id: { prefix: "TK2026-", suffix: "", width: 3, step: 1, start: 1 },
    };
    // First 3 rows already in sequence; a new batch (restarted at 1) appended.
    const rows = [
      { id: "TK2026-001" },
      { id: "TK2026-002" },
      { id: "TK2026-003" },
      { id: "TK2026-001" },
      { id: "TK2026-002" },
    ];
    applyIdPatterns(rows, pattern, 3);
    expect(rows.map((r) => r.id)).toEqual([
      "TK2026-001",
      "TK2026-002",
      "TK2026-003",
      "TK2026-004",
      "TK2026-005",
    ]);
  });

  it("respects start, step, and grows the width when the number overflows the pad", () => {
    const pattern: Record<string, IdPattern> = {
      id: { prefix: "G", suffix: "", width: 3, step: 1, start: 99 },
    };
    const rows = Array.from({ length: 3 }, () => ({ id: "" }));
    applyIdPatterns(rows, pattern);
    expect(rows.map((r) => r.id)).toEqual(["G099", "G100", "G101"]);
  });
});
