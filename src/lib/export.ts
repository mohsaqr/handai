/**
 * Shared CSV/XLSX download utility.
 *
 * Builds a blob URL and triggers an anchor-click download in the browser.
 */
import * as XLSX from "xlsx";

export async function downloadXLSX(rows: Record<string, unknown>[], filename: string): Promise<void> {
  if (!rows.length) return;
  const ws = XLSX.utils.json_to_sheet(rows);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Data");
  const fname = filename.endsWith(".xlsx") ? filename : `${filename}.xlsx`;
  XLSX.writeFile(wb, fname);
}

export async function downloadCSV(rows: Record<string, unknown>[], filename: string): Promise<void> {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(","),
    ...rows.map((r) =>
      headers
        .map((h) => `"${String(r[h] ?? "").replace(/"/g, '""')}"`)
        .join(",")
    ),
  ].join("\n");
  const content = "\uFEFF" + csv;

  // Standard blob URL download
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
