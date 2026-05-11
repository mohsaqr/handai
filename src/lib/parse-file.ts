import Papa from "papaparse";
import * as XLSX from "xlsx";
import { parseRis } from "@/lib/ris-parser";

type Row = Record<string, unknown>;

export function getFileExt(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

export function isStructuredExt(ext: string): boolean {
  return ext === "csv" || ext === "xlsx" || ext === "xls" || ext === "json" || ext === "ris";
}

/** UTF-8 decode with BOM stripped — JSON.parse refuses a leading BOM. */
function decodeUtf8(data: ArrayBuffer | Uint8Array): string {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const hasBom = bytes[0] === 0xEF && bytes[1] === 0xBB && bytes[2] === 0xBF;
  return new TextDecoder("utf-8").decode(hasBom ? bytes.subarray(3) : bytes);
}

/** Same as parseStructuredFile but works from a buffer — usable server-side
 * where the file arrives as base64, not as a File object. */
export function parseStructuredBuffer(data: ArrayBuffer | Uint8Array, ext: string): Row[] | null {
  try {
    if (ext === "csv") {
      return Papa.parse<Row>(decodeUtf8(data), { header: true, skipEmptyLines: true }).data;
    }
    if (ext === "xlsx" || ext === "xls") {
      const buf = data instanceof Uint8Array ? data : new Uint8Array(data);
      const wb = XLSX.read(buf, { type: "array" });
      const sheetsWithRows = wb.SheetNames
        .map((name) => ({ name, rows: XLSX.utils.sheet_to_json<Row>(wb.Sheets[name]) }))
        .filter((s) => s.rows.length > 0);
      if (sheetsWithRows.length === 0) return [];
      if (sheetsWithRows.length === 1) return sheetsWithRows[0].rows;
      // Multiple non-empty sheets: concatenate and tag each row with its source
      // sheet name so downstream code (and the LLM) can tell them apart. Pick
      // the shortest column name that doesn't collide with the data.
      const hasKey = (k: string) => sheetsWithRows.some((s) => s.rows.some((r) => k in r));
      let sheetCol = "sheet";
      if (hasKey(sheetCol)) {
        sheetCol = "_sheet";
        let n = 1;
        while (hasKey(sheetCol)) sheetCol = `_sheet_${n++}`;
      }
      const merged: Row[] = [];
      for (const s of sheetsWithRows) {
        for (const r of s.rows) merged.push({ [sheetCol]: s.name, ...r });
      }
      return merged;
    }
    if (ext === "json") {
      const parsed: unknown = JSON.parse(decodeUtf8(data));
      if (Array.isArray(parsed) && parsed.every((v) => v !== null && typeof v === "object" && !Array.isArray(v))) {
        return parsed as Row[];
      }
      return null;
    }
    if (ext === "ris") {
      const rows = parseRis(decodeUtf8(data));
      return rows.length > 0 ? rows : null;
    }
  } catch {
    return null;
  }
  return null;
}

/** Parses a structured file (CSV/XLSX/XLS/JSON/RIS) into rows. Returns null for
 * unstructured extensions, parse failures, or JSON that isn't an array of objects. */
export async function parseStructuredFile(file: File): Promise<Row[] | null> {
  const ext = getFileExt(file.name);
  if (!isStructuredExt(ext)) return null;
  try {
    const buf = await file.arrayBuffer();
    return parseStructuredBuffer(buf, ext);
  } catch {
    return null;
  }
}
