"use client";

import React, { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { parseStructuredFile, getFileExt } from "@/lib/parse-file";
import { useFilesRef, useFileStatuses, fileKey } from "@/hooks/useFilesRef";
import { useDropzone } from "react-dropzone";
// DataTable and ExportDropdown are used internally by ResultsPanel
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PromptEditor } from "@/components/tools/PromptEditor";
import { Input } from "@/components/ui/input";
import { useActiveModel, useSystemSettings } from "@/lib/hooks";
import { NoModelWarning } from "@/components/tools/NoModelWarning";
import { AIInstructionsSection } from "@/components/tools/AIInstructionsSection";
import { useAIInstructions, AI_INSTRUCTIONS_MARKER } from "@/hooks/useAIInstructions";
import { useSessionState, clearSessionKeys } from "@/hooks/useSessionState";
import { useFileStatesState } from "@/hooks/useFileStatesState";
import { useBatchProcessor } from "@/hooks/useBatchProcessor";
import { useRestoreSession } from "@/hooks/useRestoreSession";
import { useProcessingStore } from "@/lib/processing-store";
import { ExecutionPanel } from "@/components/tools/ExecutionPanel";
import { ResultsPanel } from "@/components/tools/ResultsPanel";
import {
  FileText,
  Upload,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Plus,
  Sparkles,
  Trash2,
  RotateCcw,
  ArrowUp,
  ArrowDown,
  Download,
  ClipboardPaste,
  Pencil,
} from "lucide-react";
import { toast } from "sonner";
import type { FieldDef, FileState } from "@/types";
import { dispatchDocumentExtract, dispatchDocumentAnalyze } from "@/lib/llm-dispatch";
import { getPrompt, formatExtractionSchema } from "@/lib/prompts";
import { Textarea } from "@/components/ui/textarea";
import { LARGE_FILE_BYTES, isLikelyChunked, distributeBudget } from "@/lib/chunk-text";
import * as XLSX from "xlsx";
import pLimit from "p-limit";

// ─── Constants ────────────────────────────────────────────────────────────────

type Row = Record<string, unknown>;

const FIELD_TYPES: FieldDef["type"][] = ["text", "number"];

const SAMPLE_EXTRACTION_PROMPTS: Record<string, string> = {
  "Invoice details": "Extract invoice details: invoice number, date, vendor name, line items with quantities and prices, and total amount.",
  "Meeting minutes": "Extract meeting minutes: date, attendees, agenda items, decisions made, action items with owners and due dates.",
  "Research findings": "Extract research findings: research question, methodology, key results, conclusions, and limitations.",
  "Contract key terms": "Extract contract key terms: parties involved, obligations, payment terms, termination conditions, and important clauses.",
  "Resume / CV data": "Extract candidate information: name, contact details, education history, work experience, and skills.",
};

function getFileTypeKey(file: File): string | null {
  const name = file.name.toLowerCase();
  const exts: Record<string, string[]> = {
    txt_md: [".txt", ".md"], pdf: [".pdf"], docx: [".docx"],
    excel: [".xlsx", ".xls"], json_csv: [".json", ".csv"], html: [".html", ".htm"],
  };
  for (const [, extList] of Object.entries(exts)) {
    if (extList.some((ext) => name.endsWith(ext))) return "supported";
  }
  return null;
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function ExtractDataPage() {
  const activeModel = useActiveModel();
  const systemSettings = useSystemSettings();

  // ── Section 1: Documents
  const filesRef = useFilesRef("extract-data");
  const [fileStates, setFileStates] = useFileStatesState("extractdata_fileStates", filesRef.current);

  // ── Section 2: Describe Data
  const [customPrompt, setCustomPrompt] = useSessionState("extractdata_customPrompt", "");

  // ── Section 3: Define Columns
  const [fields, setFields] = useSessionState<FieldDef[]>("extractdata_fields", [
    { name: "", type: "text", description: "" },
    { name: "", type: "text", description: "" },
    { name: "", type: "text", description: "" },
  ]);
  const [analyzing, setAnalyzing] = useState(false);
  const [hasSuggestedOnce, setHasSuggestedOnce] = useSessionState("extractdata_hasSuggestedOnce", false);
  const [columnMode, setColumnMode] = useState<"suggest" | "file" | "paste">("suggest");
  const [fileExtracted, setFileExtracted] = useState(false);
  const [pasteExtracted, setPasteExtracted] = useState(false);
  const [csvPasteText, setCsvPasteText] = useState("");

  // ── Column helpers ──
  const updateField = useCallback((idx: number, updates: Partial<FieldDef>) => {
    setFields((prev) => prev.map((f, i) => (i === idx ? { ...f, ...updates } : f)));
  }, [setFields]);

  const removeField = useCallback((idx: number) => {
    setFields((prev) => prev.filter((_, i) => i !== idx));
  }, [setFields]);

  const addField = useCallback(() => {
    setFields((prev) => [...prev, { name: "", type: "text", description: "" }]);
  }, [setFields]);

  const moveField = useCallback((idx: number, dir: -1 | 1) => {
    setFields((prev) => {
      const next = [...prev];
      const target = idx + dir;
      if (target < 0 || target >= next.length) return prev;
      [next[idx], next[target]] = [next[target], next[idx]];
      return next;
    });
  }, [setFields]);

  // ── Import from CSV/Excel ───────────────────────────────────────────────────
  const templateFileInputRef = useRef<HTMLInputElement>(null);
  const openTemplateFilePicker = useCallback(() => {
    templateFileInputRef.current?.click();
  }, []);
  const handleTemplateFile = useCallback((rows: Record<string, unknown>[]) => {
    if (rows.length === 0) return toast.error("File appears empty");
    const keys = Object.keys(rows[0]);
    const nameCol = keys.find((k) => /^(column_?name|name|field|col)/i.test(k)) || keys[0];
    const typeCol = keys.find((k) => /^type/i.test(k));
    const descCol = keys.find((k) => /^desc/i.test(k));
    const imported: FieldDef[] = rows.map((r) => ({
      name: String(r[nameCol] ?? "").trim(),
      type: (typeCol && FIELD_TYPES.includes(String(r[typeCol]).toLowerCase() as FieldDef["type"])
        ? String(r[typeCol]).toLowerCase() as FieldDef["type"]
        : "text"),
      description: descCol ? String(r[descCol] ?? "") : "",
    })).filter((f) => f.name);
    if (imported.length === 0) return toast.error("No columns found in file");
    setFields(imported);
    setFileExtracted(true);
    toast.success(`${imported.length} columns imported`);
  }, [setFields]);

  const handleTemplateFileSelected = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const rows = await parseStructuredFile(file);
      if (!rows) {
        toast.error(`Failed to parse .${getFileExt(file.name)} file`);
        return;
      }
      setColumnMode("file");
      handleTemplateFile(rows);
    } catch (err: unknown) {
      toast.error(`Failed to process file: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [handleTemplateFile]);

  // ── Extract from pasted CSV ───────────────────────────────────────────────
  const extractFromPastedCsv = useCallback(() => {
    const lines = csvPasteText.trim().split("\n").filter((l) => l.trim());
    if (lines.length === 0) return toast.error("No content to parse");
    const parsed: FieldDef[] = [];
    for (const line of lines) {
      const parts = line.split(",").map((s) => s.trim());
      if (parts.length === 0 || !parts[0]) continue;
      const name = parts[0];
      const rawType = (parts[1] || "").toLowerCase();
      const type = FIELD_TYPES.includes(rawType as FieldDef["type"]) ? rawType as FieldDef["type"] : "text";
      const description = parts.slice(2).join(", ");
      parsed.push({ name, type, description });
    }
    if (parsed.length === 0) return toast.error("No columns found. Use format: column_name, type, description");
    setFields(parsed);
    setCsvPasteText("");
    setPasteExtracted(true);
    toast.success(`${parsed.length} columns extracted`);
  }, [csvPasteText, setFields]);

  // ── Auto-generate AI Instructions ──────────────────────────────────────────
  const buildAutoInstructions = useCallback(() => {
    const lines: string[] = [];
    lines.push("You are a document data extractor. Extract structured information from documents.");
    lines.push("");

    if (customPrompt.trim()) {
      lines.push("EXTRACTION DESCRIPTION:");
      lines.push(customPrompt.trim());
      lines.push("");
    }

    const namedFields = fields.filter((f) => f.name.trim());
    if (namedFields.length > 0) {
      lines.push("FIELDS TO EXTRACT:");
      namedFields.forEach((f) => {
        lines.push(`- ${f.name} (${f.type})${f.description ? `: ${f.description}` : ""}`);
      });
      lines.push("");
    }

    lines.push("RULES:");
    lines.push("- Extract data from the document content");
    lines.push("- Return a JSON object with the defined field names as keys");
    lines.push("- If a field cannot be found, return null for that field");
    lines.push("- Do not include markdown or code fences");
    lines.push("");
    lines.push(AI_INSTRUCTIONS_MARKER);

    return lines.join("\n");
  }, [customPrompt, fields]);

  // ── Section 4: AI Instructions
  const [aiInstructions, setAiInstructions] = useAIInstructions(buildAutoInstructions);

  // ── System prompt ──────────────────────────────────────────────────────────
  const buildSystemPrompt = (): string => {
    if (aiInstructions.trim()) return aiInstructions;
    const namedFields = fields.filter((f) => f.name.trim());
    if (namedFields.length > 0) {
      return getPrompt("document.extraction").replace("{schema}", formatExtractionSchema(namedFields));
    }
    return (
      customPrompt.trim() ||
      getPrompt("document.extraction").replace(
        "{schema}",
        "(no schema defined — extract all logical records with appropriate column names)"
      )
    );
  };

  // ── File drop ──────────────────────────────────────────────────────────────
  const onDrop = useCallback(
    (accepted: File[]) => {
      const valid = accepted.filter((f) => getFileTypeKey(f) !== null);
      const skipped = accepted.length - valid.length;
      if (skipped > 0) toast.warning(`${skipped} file(s) skipped — unsupported type`);
      valid.forEach((f) => filesRef.current.set(fileKey(f), f));
      setFileStates((prev) => [
        ...prev,
        ...valid.map((f): FileState => ({ file: f, status: "pending" })),
      ]);
    },
    [setFileStates, filesRef]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, multiple: true,
  });

  const removeFile = (idx: number) => {
    const fs = fileStates[idx];
    if (fs) filesRef.current.delete(fileKey(fs.file));
    setFileStates((prev) => prev.filter((_, i) => i !== idx));
  };

  // ── AI Suggest ─────────────────────────────────────────────────────────────
  const suggestFields = async () => {
    if (fileStates.length === 0) return toast.error("Upload at least one file first");
    if (!activeModel) return toast.error("No model configured. Add an API key in Settings.");

    setAnalyzing(true);
    try {
      // Extract text from every file in parallel, then share a fixed total
      // character budget across them: each file gets `TOTAL_BUDGET / N`, and
      // any leftover from short files is redistributed to longer ones.
      const { extractTextBrowser } = await import("@/lib/document-browser");
      const TOTAL_BUDGET = 3000;
      const limit = pLimit(systemSettings.maxConcurrency || 5);
      const settledTexts = await Promise.allSettled(
        fileStates.map((fs) => limit(async () => {
          if (fs.file.size === 0) return { name: fs.file.name, text: "" };
          const { text } = await extractTextBrowser(fs.file);
          return { name: fs.file.name, text };
        }))
      );

      let failed = 0;
      const fullSections: Array<{ name: string; text: string }> = [];
      for (const r of settledTexts) {
        if (r.status !== "fulfilled" || !r.value.text.trim()) { failed++; continue; }
        fullSections.push(r.value);
      }
      if (fullSections.length === 0) {
        toast.error("Could not read any of the uploaded files.");
        return;
      }

      // Allocate the shared budget proportionally across the surviving files.
      const allocations = distributeBudget(fullSections.map((s) => s.text.length), TOTAL_BUDGET);
      const readSections = fullSections.map((s, i) => ({ name: s.name, text: s.text.slice(0, allocations[i]) }));

      // Alphabetize so upload order doesn't bias the suggestions.
      readSections.sort((a, b) => a.name.localeCompare(b.name));
      const sectionBlocks = readSections.map((s) => `=== ${s.name} ===\n\n${s.text}`);
      const preamble =
        `The text below contains ${readSections.length} separate document${readSections.length !== 1 ? "s" : ""}, ` +
        `each introduced by a "=== filename ===" header and separated by "---". ` +
        `Treat every section as equally important — do NOT prefer fields from earlier sections over later ones. ` +
        `Derive a single unified schema that covers all documents.`;
      const combinedText = `${preamble}\n\n${sectionBlocks.join("\n\n---\n\n")}`;
      const combinedFile = new File([combinedText], "combined_documents.txt", { type: "text/plain" });

      const { fields: suggested } = await dispatchDocumentAnalyze({
        file: combinedFile,
        provider: activeModel.providerId,
        model: activeModel.defaultModel,
        apiKey: activeModel.apiKey || "",
        baseUrl: activeModel.baseUrl,
        hint: customPrompt.trim() || undefined,
      });

      // Dedupe by lowercased name (LLM may still repeat similar names across sections).
      const validTypes = new Set(FIELD_TYPES);
      const merged = new Map<string, FieldDef>();
      for (const f of (suggested ?? [])) {
        const key = (f.name || "").trim().toLowerCase();
        if (!key || merged.has(key)) continue;
        merged.set(key, {
          name: f.name,
          type: validTypes.has(f.type) ? f.type : "text",
          description: f.description || "",
        });
      }

      if (merged.size > 0) {
        setFields(Array.from(merged.values()));
        setHasSuggestedOnce(true);
        const analyzed = fileStates.length - failed;
        toast.success(`${merged.size} fields suggested from ${analyzed} file${analyzed !== 1 ? "s" : ""}${failed > 0 ? ` (${failed} skipped)` : ""}`);
      } else {
        toast.info("No suggestions returned. Try with more structured documents.");
      }
    } catch (err: unknown) {
      toast.error("Analysis failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setAnalyzing(false);
    }
  };

  // ── Build data rows from files (one row per file for useBatchProcessor) ────
  const data: Row[] = useMemo(() =>
    fileStates.map((fs, i) => ({
      _fileIdx: i,
      document_name: fs.file.name,
      _fileKey: fileKey(fs.file),
    })),
    [fileStates]
  );

  // ── Batch processor ───────────────────────────────────────────────────────
  const batch = useBatchProcessor({
    toolId: "/extract-data",
    runType: "extract-data",
    activeModel,
    systemSettings,
    data,
    dataName: fileStates.map((f) => f.file.name).join(", ") || "unnamed",
    systemPrompt: aiInstructions || buildSystemPrompt(),
    validate: () => {
      if (fileStates.length === 0) return "Upload at least one file";
      return null;
    },
    selectData: (_data: Row[], mode) => {
      return mode === "test" ? _data.slice(0, 1) : _data;
    },
    processRow: async (row: Row) => {
      const fKey = row._fileKey as string;
      const file = filesRef.current.get(fKey);
      if (!file) throw new Error(`File not found: ${row.document_name}`);

      // Snapshot content for restorable file types so a future "Restore Session"
      // can rebuild a real, runnable File. Binary types (PDF/DOCX) skip this and
      // stay as placeholders on restore.
      const RESTORE_SIZE_CAP = 5_000_000; // 5 MB — beyond this, fall back to placeholder
      const ext = getFileExt(file.name);
      let restoreContent: string | undefined;
      let restoreMime: string | undefined;
      let restoreName: string | undefined;
      try {
        if (file.size <= RESTORE_SIZE_CAP) {
          if (["csv", "json", "txt", "md", "html", "htm"].includes(ext)) {
            restoreContent = await file.text();
            restoreMime = file.type || "text/plain";
            restoreName = file.name;
          } else if (ext === "xlsx" || ext === "xls") {
            const buf = await file.arrayBuffer();
            const wb = XLSX.read(buf, { type: "array" });
            const firstSheet = wb.Sheets[wb.SheetNames[0]];
            restoreContent = XLSX.utils.sheet_to_csv(firstSheet);
            restoreMime = "text/csv";
            restoreName = file.name.replace(/\.(xlsx|xls)$/i, ".csv");
          }
        }
      } catch {
        // Snapshot is best-effort — failure just means restore falls back to placeholder.
      }

      const systemPrompt = buildSystemPrompt();
      const namedFields = fields.filter((f) => f.name.trim());

      const t0 = Date.now();
      const result = await dispatchDocumentExtract({
        file,
        provider: activeModel!.providerId,
        model: activeModel!.defaultModel,
        apiKey: activeModel!.apiKey || "",
        baseUrl: activeModel!.baseUrl,
        systemPrompt,
        fields: namedFields.length > 0 ? namedFields : undefined,
        maxTokens: systemSettings.maxTokens ?? undefined,
      });

      const latency = Date.now() - t0;
      if (result.chunks > 1) {
        const msg = result.failedChunks > 0
          ? `Processed ${file.name} in ${result.chunks} sections (${result.failedChunks} failed)`
          : `Processed ${file.name} in ${result.chunks} sections`;
        toast.info(msg);
      }
      return {
        document_name: file.name,
        _all_records: JSON.stringify(result.records ?? []),
        _record_count: result.count,
        _chunk_count: result.chunks,
        status: "success",
        latency_ms: latency,
        ...(restoreContent !== undefined && {
          _file_content: restoreContent,
          _file_mime: restoreMime,
          _file_restore_name: restoreName,
        }),
      };
    },
    buildResultEntry: (r: Row, i: number) => ({
      rowIndex: i,
      input: {
        document_name: r.document_name,
        ...(typeof r._file_content === "string" && {
          _file_content: r._file_content,
          _file_mime: r._file_mime,
          _file_restore_name: r._file_restore_name,
        }),
      } as Record<string, unknown>,
      output: (r._all_records as string) ?? JSON.stringify(r),
      status: (r.status as string) ?? "success",
      latency: r.latency_ms as number | undefined,
      errorMessage: r.error_msg as string | undefined,
    }),
    onComplete: () => {},
  });

  const fileStatuses = useFileStatuses(fileStates, batch.results);

  // ── Build flat table from all records ─────────────────────────────────────
  const allResults: Row[] = useMemo(() => {
    const rows: Row[] = [];
    for (const r of batch.results) {
      if (r.status !== "success" || !r._all_records) continue;
      try {
        let records = JSON.parse(r._all_records as string) as Row[];
        // Detect the "fragment" pattern where the LLM splits ONE record across
        // multiple single-field objects (e.g. [{full_name: "..."}, {birth_year: "..."}])
        // and merge them back into one record. We must NOT collapse legitimate
        // single-column multi-record output (e.g. [{name: "Alice"}, {name: "Bob"}]) —
        // that's only true fragmentation when the keys across records are DISTINCT.
        if (records.length > 1 && records.every((rec) => Object.keys(rec).length === 1)) {
          const allKeys = new Set(records.flatMap((rec) => Object.keys(rec)));
          if (allKeys.size === records.length) {
            const merged: Row = {};
            for (const rec of records) Object.assign(merged, rec);
            records = [merged];
          }
        }
        for (const rec of records) {
          // Clean record: remove keys that are clearly not field data
          const clean: Row = { document_name: r.document_name };
          for (const [k, v] of Object.entries(rec)) {
            if (k && k.length > 1 && !k.startsWith("_") && v !== undefined) {
              clean[k] = v;
            }
          }
          clean.status = r.status ?? "success";
          clean.latency_ms = typeof r.latency_ms === "number" ? r.latency_ms : 0;
          rows.push(clean);
        }
      } catch {
        // skip unparseable
      }
    }
    return rows;
  }, [batch.results]);

  // ── Session restore from history ───────────────────────────────────────────
  const restored = useRestoreSession("extract-data");
  useEffect(() => {
    if (!restored) return;
    queueMicrotask(() => {
      const fullPrompt = restored.systemPrompt ?? "";

      // Restore extraction prompt
      const descMatch = fullPrompt.match(/EXTRACTION DESCRIPTION:\n([\s\S]*?)(?:\n\n|$)/);
      if (descMatch) setCustomPrompt(descMatch[1].trim());

      // Restore fields from system prompt
      const fieldsMatch = fullPrompt.match(/FIELDS TO EXTRACT:\n([\s\S]*?)(?:\n\n|$)/);
      if (fieldsMatch) {
        const parsed: FieldDef[] = fieldsMatch[1]
          .split("\n")
          .map((l) => l.replace(/^- /, "").trim())
          .filter(Boolean)
          .map((l) => {
            const m = l.match(/^(.+?)\s*\((\w+)\)(?::\s*(.*))?$/);
            if (!m) return { name: l, type: "text" as const, description: "" };
            return {
              name: m[1].trim(),
              type: FIELD_TYPES.includes(m[2] as FieldDef["type"]) ? m[2] as FieldDef["type"] : "text" as const,
              description: m[3]?.trim() || "",
            };
          });
        if (parsed.length > 0) setFields(parsed);
      }

      // Restore file list. For text-based / spreadsheet inputs (CSV/JSON/TXT/MD/
      // HTML/XLSX→CSV), processRow snapshotted the content into inputJson, so we
      // can rebuild a real, runnable File. Binary types (PDF/DOCX) fall through
      // to a zero-byte placeholder — user must re-upload to re-run.
      const restoredFiles: FileState[] = restored.data.map((row) => {
        const name = (row.document_name as string) || "document";
        const content = row._file_content;
        if (typeof content === "string") {
          const restoreName = (row._file_restore_name as string) || name;
          const mime = (row._file_mime as string) || "text/plain";
          const real = new File([content], restoreName, { type: mime });
          filesRef.current.set(fileKey(real), real);
          return { file: real, status: "done" as const };
        }
        const placeholder = new File([], name);
        filesRef.current.set(fileKey(placeholder), placeholder);
        return { file: placeholder, status: "done" as const };
      });
      setFileStates(restoredFiles);

      // Populate results in global processing store
      const errors = restored.results.filter((r) => r.status === "error").length;
      useProcessingStore.getState().completeJob(
        "/extract-data",
        restored.results,
        { success: restored.results.length - errors, errors, avgLatency: 0 },
        restored.runId,
      );
      // restored.data has one entry per processed file (extract-data saves
      // `{document_name}` as inputJson). The actual extracted-record count
      // lives inside each row's `_all_records` JSON array — count those for
      // the toast so it matches what the result table shows.
      const fileCount = restored.results.length;
      const recordCount = restored.results.reduce((sum, r) => {
        const allRecords = r._all_records;
        if (typeof allRecords !== "string") return sum;
        try {
          const parsed = JSON.parse(allRecords);
          return sum + (Array.isArray(parsed) ? parsed.length : 1);
        } catch {
          return sum;
        }
      }, 0);
      toast.success(
        `Restored session from "${restored.dataName}" (${recordCount} record${recordCount === 1 ? "" : "s"} from ${fileCount} file${fileCount === 1 ? "" : "s"})`
      );
    });
  }, [restored]);

  const successFiles = batch.results.filter((r) => r.status === "success").length;
  const totalChunks = batch.results.reduce((sum, r) => sum + (Number(r._chunk_count) || 1), 0);
  const chunkNote = totalChunks > successFiles ? ` (${totalChunks} sections)` : "";
  const extractSubtitle = `${allResults.length} records from ${successFiles} file(s)${chunkNote}`;

  const renderSchemaTable = () => (
    <div className="border rounded-lg overflow-hidden">
      <div className="px-4 py-2.5 border-b bg-muted/20 text-sm font-medium">Column Schema</div>
      <div className="px-3 pt-2 flex gap-2 items-center text-xs font-medium text-muted-foreground">
        <div className="shrink-0 w-6" />
        <div className="flex-1">column_name</div>
        <div className="w-28">type</div>
        <div className="flex-1">description</div>
        <div className="w-8 shrink-0" />
      </div>
      <div className="p-3 space-y-2">
        {fields.map((field, idx) => (
          <div key={idx} className="flex gap-2 items-center">
            <div className="flex flex-col shrink-0">
              <Button variant="ghost" size="icon" className="h-4 w-6 text-muted-foreground hover:text-foreground" onClick={() => moveField(idx, -1)} disabled={idx === 0}>
                <ArrowUp className="h-3 w-3" />
              </Button>
              <Button variant="ghost" size="icon" className="h-4 w-6 text-muted-foreground hover:text-foreground" onClick={() => moveField(idx, 1)} disabled={idx === fields.length - 1}>
                <ArrowDown className="h-3 w-3" />
              </Button>
            </div>
            <Input placeholder="column_name" value={field.name} onChange={(e) => updateField(idx, { name: e.target.value })} className="flex-1 h-8 text-xs" />
            <Select value={field.type} onValueChange={(v) => updateField(idx, { type: v as FieldDef["type"] })}>
              <SelectTrigger className="w-28 h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {FIELD_TYPES.map((t) => (<SelectItem key={t} value={t} className="text-xs">{t}</SelectItem>))}
              </SelectContent>
            </Select>
            <Input placeholder="Description (optional)" value={field.description || ""} onChange={(e) => updateField(idx, { description: e.target.value })} className="flex-1 h-8 text-xs" />
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-destructive shrink-0" onClick={() => removeField(idx)}>
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        ))}
      </div>
      <div className="px-3 pb-3 flex gap-2">
        <Button variant="outline" size="sm" className="flex-1 text-xs" onClick={addField}>
          <Plus className="h-3 w-3 mr-2" /> Add Column
        </Button>
        <Button variant="outline" size="sm" className="text-xs text-destructive hover:bg-destructive/10" onClick={() => setFields([{ name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }])}>
          <Trash2 className="h-3 w-3 mr-2" /> Clear All
        </Button>
      </div>
    </div>
  );

  return (
    <div className="space-y-0 pb-16">

      {/* Header */}
      <div className="pb-6 flex items-start justify-between">
        <div className="space-y-1 max-w-3xl">
          <h1 className="text-4xl font-bold">Extract Data</h1>
          <p className="text-muted-foreground text-sm">
            Extract structured tabular data from documents using AI
          </p>
        </div>
        <Button variant="destructive" className="gap-2 px-5" onClick={() => { clearSessionKeys("extractdata_"); batch.clearResults(); filesRef.current.clear(); setFileStates([]); setCustomPrompt(""); setFields([{ name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }]); setHasSuggestedOnce(false); setAiInstructions(""); setColumnMode("suggest"); setFileExtracted(false); setPasteExtracted(false); setCsvPasteText(""); }}>
            <RotateCcw className="h-3.5 w-3.5" /> Start Over
          </Button>
      </div>

      <div className={batch.isProcessing ? "pointer-events-none opacity-60" : ""}>
      {/* ── 1. Upload Documents ─────────────────────────────────────────── */}
      <div className="space-y-4 pb-8">
        <h2 className="text-2xl font-bold">1. Upload Documents</h2>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/30 hover:border-primary/50 hover:bg-muted/20"
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="p-3 rounded-full bg-muted">
              <Upload className="h-6 w-6 text-muted-foreground" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium">
                {isDragActive ? "Drop the files here" : "Click or drag files to upload"}
              </p>
              <p className="text-xs text-muted-foreground">
                PDF, DOCX, Excel, TXT, MD, JSON, CSV, HTML
              </p>
            </div>
          </div>
        </div>

        {fileStates.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between px-1">
              <span className="text-xs text-muted-foreground">{fileStates.length} file{fileStates.length !== 1 ? "s" : ""}</span>
              <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground" onClick={() => { filesRef.current.clear(); setFileStates([]); batch.clearResults(); toast.success("Cleared all files"); }}>
                <Trash2 className="h-3 w-3 mr-1" /> Clear All
              </Button>
            </div>
            {fileStates.map((entry, idx) => {
              const status = batch.isProcessing || batch.results.length > 0 ? fileStatuses[idx] : entry.status;
              const resultRow = batch.results[idx];
              const recordCount = resultRow?._record_count as number | undefined;
              const errorMsg = resultRow?.error_msg as string | undefined;
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg border bg-muted/20">
                    <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate text-xs">{entry.file.name}</span>
                    {entry.file.size === 0 ? (
                      <span className="text-[10px] text-red-600 dark:text-red-400 shrink-0 italic" title="Restored placeholder — original file contents are not stored. Re-upload the file to re-run.">
                        Placeholder · re-upload to re-run
                      </span>
                    ) : (
                      <span className="text-[10px] text-muted-foreground shrink-0">
                        {(entry.file.size / 1024).toFixed(0)} KB
                      </span>
                    )}

                    {status === "pending" && isLikelyChunked(entry.file.size) && (
                      <span className="text-[10px] text-amber-600 dark:text-amber-400 shrink-0" title="This file will be split into sections for complete extraction">
                        Multi-section
                      </span>
                    )}
                    {status === "pending" && !isLikelyChunked(entry.file.size) && (
                      <span className="text-[10px] text-muted-foreground shrink-0">Pending</span>
                    )}
                    {(status === "extracting" || status === "analyzing") && (
                      <span className="flex items-center gap-1 text-[10px] text-purple-600 shrink-0">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Analyzing
                      </span>
                    )}
                    {status === "done" && (
                      <span className="flex items-center gap-1 text-[10px] text-green-600 shrink-0">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        {recordCount ?? 0} records
                      </span>
                    )}
                    {status === "error" && (
                      <span className="flex items-center gap-1 text-[10px] text-red-500 shrink-0" title={errorMsg}>
                        <AlertCircle className="h-3.5 w-3.5" /> Error
                      </span>
                    )}

                    <button onClick={() => removeFile(idx)} className="text-muted-foreground hover:text-destructive shrink-0">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {status === "error" && errorMsg && (
                    <div className="ml-3 text-[10px] text-red-500 leading-snug">{errorMsg}</div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="border-t" />

      {/* ── 2. Extraction Prompt ─────────────────────────────────────────── */}
      <div className="space-y-3 py-8">
        <h2 className="text-2xl font-bold">2. Extraction Prompt</h2>
        <PromptEditor
          value={customPrompt}
          onChange={setCustomPrompt}
          placeholder="Describe what you want to extract from the documents. E.g.: Extract invoice details including amounts, dates, and vendor names..."
          examplePrompts={SAMPLE_EXTRACTION_PROMPTS}
          label="Instructions"
          helpText="Describe the data you want to extract. This feeds into the AI instructions and helps the AI suggest columns."
        />
      </div>

      <div className="border-t" />

      {/* ── 3. Define Columns ──────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">3. Define Columns</h2>
        <p className="text-sm text-muted-foreground -mt-2">
          Define your output columns. Use AI to suggest columns from your document, or add them manually.
        </p>

        <div className="flex gap-2 flex-wrap">
          <Button variant={columnMode === "suggest" ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setColumnMode("suggest")}>
            <Sparkles className="h-3.5 w-3.5 mr-1.5" /> Suggest with AI
          </Button>
          <Button variant={columnMode === "paste" ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setColumnMode("paste")}>
            <ClipboardPaste className="h-3.5 w-3.5 mr-1.5" /> Type CSV
          </Button>
          <Button variant={columnMode === "file" ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setColumnMode("file")}>
            <Upload className="h-3.5 w-3.5 mr-1.5" /> Import CSV/Excel
          </Button>
          <Button variant="outline" size="sm" className="text-xs" onClick={() => {
            const named = fields.filter((f) => f.name.trim());
            const rows = named.length > 0
              ? named.map((f) => ({ column_name: f.name, type: f.type, description: f.description || "" }))
              : [{ column_name: "", type: "", description: "" }];
            const ws = XLSX.utils.json_to_sheet(rows, { header: ["column_name", "type", "description"] });
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, "Schema");
            XLSX.writeFile(wb, "column_schema.xlsx");
          }}>
            <Download className="h-3.5 w-3.5 mr-1.5" /> Export Excel
          </Button>
        </div>

        {/* Hidden file input for Import mode */}
        <input
          ref={templateFileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={handleTemplateFileSelected}
        />

        {/* ── Suggest with AI mode ── */}
        {columnMode === "suggest" && (
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="text-xs" onClick={suggestFields} disabled={analyzing}>
              {analyzing ? <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5 mr-1.5" />}
              {hasSuggestedOnce ? "Retry AI" : "Ask AI"}
            </Button>
            {analyzing && <span className="text-xs text-muted-foreground">Analyzing document for field suggestions...</span>}
          </div>
        )}

        {/* ── Import CSV/Excel mode ── */}
        {columnMode === "file" && (
          <Button variant="outline" size="sm" className="text-xs" onClick={openTemplateFilePicker}>
            <Upload className="h-3.5 w-3.5 mr-1.5" /> {fileExtracted ? "Re-import" : "Choose File"}
          </Button>
        )}

        {/* ── Type CSV mode ── */}
        {columnMode === "paste" && !pasteExtracted && !fields.some((f) => f.name.trim()) && (
          <div className="space-y-2">
            <Textarea
              placeholder={"One column per line: column_name, type, description\n\nauthor_name, text, full name of the author\npublication_year, number, year published\ntopic, text, main topic"}
              className="min-h-[100px] text-xs font-mono resize-y"
              value={csvPasteText}
              onChange={(e) => setCsvPasteText(e.target.value)}
            />
            <Button variant="outline" size="sm" className="text-xs" onClick={extractFromPastedCsv}>
              Extract Columns
            </Button>
          </div>
        )}
        {columnMode === "paste" && (pasteExtracted || fields.some((f) => f.name.trim())) && (
          <Button variant="outline" size="sm" className="text-xs" onClick={() => {
            const text = fields.filter((f) => f.name.trim()).map((f) => `${f.name}, ${f.type}, ${f.description || ""}`).join("\n");
            setCsvPasteText(text);
            setPasteExtracted(false);
            setFields([{ name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }, { name: "", type: "text", description: "" }]);
          }}>
            <Pencil className="h-3.5 w-3.5 mr-1.5" /> Edit as CSV
          </Button>
        )}

        {/* ── Column Schema (hidden while editing as CSV in paste mode) ── */}
        {!(columnMode === "paste" && !pasteExtracted && !fields.some((f) => f.name.trim())) && renderSchemaTable()}
      </div>

      <div className="border-t" />

      {/* ── 4. AI Instructions ─────────────────────────────────────────── */}
      <AIInstructionsSection
        sectionNumber={4}
        value={aiInstructions}
        onChange={setAiInstructions}
      >
        <NoModelWarning activeModel={activeModel} />
      </AIInstructionsSection>

      </div>

      <div className="border-t" />

      {/* ── 5. Execute ──────────────────────────────────────────────────── */}
      <div className="space-y-4 py-8">
        <h2 className="text-2xl font-bold">5. Execute</h2>

        {fileStates.some((fs) => isLikelyChunked(fs.file.size)) && (
          <div className="flex items-start gap-3 px-4 py-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 text-sm text-amber-700 dark:text-amber-300">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              Some files are large and will be automatically split into sections for complete extraction.
              This uses additional API calls but ensures no data is missed.
            </span>
          </div>
        )}

        <ExecutionPanel
          isProcessing={batch.isProcessing}
          aborting={batch.aborting}
          runMode={batch.runMode}
          progress={batch.progress}
          etaStr={batch.etaStr}
          dataCount={fileStates.length}
          disabled={fileStates.length === 0 || !activeModel}
          onRun={batch.run}
          onAbort={batch.abort}
          onResume={batch.resume}
          onCancel={batch.clearResults}
          failedCount={batch.failedCount}
          skippedCount={batch.skippedCount}
          unitLabel="file"
          testLabel="Test (1 file)"
          fullLabel={`Process All (${fileStates.length} file${fileStates.length !== 1 ? "s" : ""})`}
        />
      </div>

      {/* ── Results ─────────────────────────────────────────────────────── */}
      <ResultsPanel
        results={allResults}
        runId={batch.runId}
        title="Extracted Data"
        subtitle={extractSubtitle}
      />
    </div>
  );
}
