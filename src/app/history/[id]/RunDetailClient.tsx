"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DataTable, ExportDropdown } from "@/components/tools/DataTable";
import {
    ArrowLeft,
    Download,
    Calendar,
    Clock,
    Cpu,
    CheckCircle2,
    AlertCircle,
    Loader2,
    Trash2,
    ChevronRight,
    RotateCcw,
    Copy,
    Check,
    Pencil,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { toast } from "sonner";
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { getRun as idbGetRun, deleteRun as idbDeleteRun, renameRun as idbRenameRun } from "@/lib/db-indexeddb";
import { useRestoreStore, type RestorePayload } from "@/lib/restore-store";
import { useBrowserStorage } from "@/lib/llm-dispatch";
import type { RunMeta, RunResult } from "@/types";

const useBrowserDb = useBrowserStorage;

export default function RunDetailClient({ id }: { id: string }) {
    const router = useRouter();
    const [run, setRun] = useState<RunMeta | null>(null);
    const [results, setResults] = useState<Record<string, unknown>[]>([]);
    const [rawResults, setRawResults] = useState<RunResult[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editName, setEditName] = useState("");
    const editRef = useRef<HTMLInputElement>(null);
    const setPendingRestore = useRestoreStore((s) => s.setPending);
    const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

    /** Parse the GENERATED CODEBOOK: block out of a codebook-generator run's
     * systemPrompt. Returns display rows for the codebook detail table. */
    const parseCodebookFromSystemPrompt = (systemPrompt: string): Record<string, unknown>[] => {
        const m = systemPrompt.match(/GENERATED CODEBOOK:\n([\s\S]*?)$/);
        if (!m) return [];
        try {
            const parsed = JSON.parse(m[1].trim()) as Record<string, unknown>[];
            return parsed.map((e) => ({
                "Code label": String(e.code ?? ""),
                Description: String(e.description ?? ""),
                Examples: String(e.example ?? ""),
            }));
        } catch {
            return [];
        }
    };

    /** Turn a single RunResult into one or more display rows. */
    const buildResultRows = (r: RunResult, runType?: string): Record<string, unknown>[] => {
        const rawInput = JSON.parse(r.inputJson ?? "{}");
        // Restore-only metadata: extract-data and process-documents snapshot input
        // file content into inputJson so "Restore Session" can rebuild real Files.
        // Hide these from the history-table display.
        // - _file_content/_file_mime/_file_restore_name: per-row snapshot (structured runs)
        // - _files: array of per-file snapshots (unified process-documents runs)
        const {
            _file_content: _fc,
            _file_mime: _fm,
            _file_restore_name: _frn,
            _files: _fs,
            ...input
        } = rawInput;
        void _fc; void _fm; void _frn; void _fs;
        const hasAiOutput = Object.keys(input).some((k) => k.startsWith("ai_output"));

        const meta = {
            status: r.status,
            latency_ms: Math.round(r.latency ?? 0),
            ...(r.errorMessage ? { error_message: r.errorMessage } : {}),
        };

        // process-documents outputs are freeform text — never parse/spread them
        if (runType === "process-documents") {
            return [{ ...input, output: r.output ?? "", ...meta }];
        }

        // model-comparison (formerly consensus-coder) stores judge_output +
        // worker_* columns inside input; r.output is intentionally blank, so
        // skip the duplicate "output" column.
        if (runType === "model-comparison" || runType === "consensus-coder") {
            return [{ ...input, ...meta }];
        }

        // agent-panel stores all per-step / per-worker / per-round outputs (and
        // the final reconciler/consensus column) inside input; r.output is
        // intentionally blank, so skip the duplicate "output" column. Reorder
        // so columns flow: originals → agent outputs → final/consensus columns
        // → per-step latencies → status → latency_ms → error_message.
        if (runType === "agent-panel") {
            const merged: Record<string, unknown> = { ...input, ...meta };
            const allKeys = Object.keys(merged);
            const TRAILING = new Set(["status", "latency_ms", "error_message"]);
            const FINAL_KEYS = new Set(["final_output", "final_consensus", "reconciler_output", "rounds_taken", "converged"]);
            const originalKeys = allKeys.filter((k) => !k.endsWith("_output") && !k.endsWith("_latency_ms") && !TRAILING.has(k) && !FINAL_KEYS.has(k));
            const outputKeys = allKeys.filter((k) => k.endsWith("_output") && !FINAL_KEYS.has(k));
            const finalKeys = allKeys.filter((k) => FINAL_KEYS.has(k));
            const latencyKeys = allKeys.filter((k) => k.endsWith("_latency_ms"));
            const trailing = ["status", "latency_ms", "error_message"].filter((k) => k in merged);
            const ordered = [...originalKeys, ...outputKeys, ...finalKeys, ...latencyKeys, ...trailing];
            const out: Record<string, unknown> = {};
            ordered.forEach((k) => { out[k] = merged[k]; });
            return [out];
        }

        // qualitative-coder stores the assigned code in r.output as a plain string,
        // but the live tool / restore path label it `ai_code`. Drop the redundant
        // `output` column from the history table — restore reads rawResults, so
        // the value itself is still preserved.
        if (runType === "qualitative-coder") {
            return [{ ...input, ...meta }];
        }

        // ai-coder: r.output is one of three shapes (mirrors parseAIResponse in
        // src/app/ai-coder/page.tsx):
        //   1) {codes: {label: pct, ...}, reasoning: "..."}     — preferred
        //   2) {label: pct, ...}                                 — legacy flat
        //   3) "Code A, Code B, Uncoded"                         — comma fallback
        // It may also be wrapped in ```json ... ``` fences. The earlier version
        // only handled (1) unfenced, leaving (2)/(3)/fenced rows empty.
        if (runType === "ai-coder") {
            const { ai_codes: _rawAi, status: _s, latency_ms: _l, ...cleanInput } = input as Record<string, unknown>;
            void _rawAi; void _s; void _l;
            let formattedCodes = "";
            let decision = "";
            let reasoning = "";
            const cleanKey = (k: string) => k.split(/\s*[—–]\s/)[0].trim();

            const buildFromConfidence = (conf: [string, number][]) => {
                const sorted = [...conf]
                    .filter(([, v]) => Number.isFinite(v) && Math.round(v) > 0)
                    .sort(([, a], [, b]) => b - a);
                formattedCodes = JSON.stringify(Object.fromEntries(
                    sorted.map(([k, v]) => [k, `${Math.round(v)}%`])
                ));
                const numericOnly = conf.filter(([, v]) => Number.isFinite(v));
                if (numericOnly.length > 0) {
                    const maxVal = Math.max(...numericOnly.map(([, v]) => v));
                    decision = numericOnly
                        .filter(([, v]) => v === maxVal)
                        .map(([k]) => k)
                        .join(", ");
                }
            };

            if (typeof r.output === "string" && r.output) {
                const stripped = r.output.replace(/^```json?\s*/i, "").replace(/\s*```$/, "").trim();
                let handled = false;
                try {
                    const parsed = JSON.parse(stripped);
                    if (parsed && typeof parsed === "object") {
                        // Shape 1: nested under `codes`
                        if (parsed.codes && typeof parsed.codes === "object" && !Array.isArray(parsed.codes)) {
                            const conf = Object.entries(parsed.codes as Record<string, unknown>).map(([k, v]) => {
                                const num = typeof v === "number" ? v : Number(v);
                                return [cleanKey(k), num] as [string, number];
                            });
                            buildFromConfidence(conf);
                            handled = true;
                        } else if (!Array.isArray(parsed)) {
                            // Shape 2: flat object of code → number
                            const conf = Object.entries(parsed as Record<string, unknown>)
                                .map(([k, v]) => {
                                    const num = typeof v === "number" ? v : Number(v);
                                    return [cleanKey(k), num] as [string, number];
                                })
                                .filter(([, v]) => Number.isFinite(v));
                            if (conf.length > 0) {
                                buildFromConfidence(conf);
                                handled = true;
                            }
                        }
                        if (typeof parsed.reasoning === "string") reasoning = parsed.reasoning;
                    }
                } catch {
                    // not JSON — fall through to comma-list fallback
                }
                if (!handled) {
                    // Shape 3: comma-separated code list
                    const tokens = stripped.split(",").map((s) => s.trim()).filter((s) => s && s !== "Uncoded");
                    if (tokens.length > 0) {
                        const conf = tokens.map((t) => [cleanKey(t), 80] as [string, number]);
                        buildFromConfidence(conf);
                    } else if (!formattedCodes) {
                        // Last resort: surface the raw output so the cell isn't blank
                        formattedCodes = stripped;
                    }
                }
            }
            return [{ ...cleanInput, ai_codes: decision, ai_probabilities: formattedCodes, ai_reasoning: reasoning, ...meta }];
        }

        // abstract-screener: input was saved with ai_decision / ai_confidence / ai_probabilities
        // (JSON string of {include,maybe,exclude} as 0-1 floats) / ai_reasoning / ai_highlight_terms
        // (JSON string of string[]) plus status/latency_ms. The generic JSON-spread path also adds
        // duplicate `decision/confidence/probabilities/reasoning` columns from r.output. Emit a
        // clean, ordered shape and drop the duplicates.
        if (runType === "abstract-screener") {
            const {
                ai_decision: aiDecision,
                ai_confidence: _conf,
                ai_probabilities: aiProbs,
                ai_reasoning: aiReasoning,
                ai_highlight_terms: aiHighlightTerms,
                status: _s,
                latency_ms: _l,
                error_msg: _em,
                ...cleanInput
            } = input as Record<string, unknown>;
            void _conf; void _s; void _l; void _em;

            // Format probabilities as a single-line JSON object, sorted desc, 0% dropped
            // (mirrors the ai-coder history view above).
            let formattedProbs = "";
            const probsObj = typeof aiProbs === "string"
                ? (() => { try { return JSON.parse(aiProbs); } catch { return null; } })()
                : (aiProbs && typeof aiProbs === "object" ? aiProbs : null);
            if (probsObj && typeof probsObj === "object") {
                const entries = Object.entries(probsObj as Record<string, unknown>)
                    .map(([k, v]) => {
                        const num = typeof v === "number" ? v : Number(v);
                        return [k, Number.isFinite(num) ? Math.round(num * 100) : NaN] as [string, number];
                    })
                    .filter(([, pct]) => Number.isFinite(pct) && pct > 0)
                    .sort(([, a], [, b]) => b - a);
                formattedProbs = JSON.stringify(Object.fromEntries(
                    entries.map(([k, pct]) => [k, `${pct}%`])
                ));
            } else if (typeof aiProbs === "string") {
                formattedProbs = aiProbs;
            }

            // Format highlight terms (stored as JSON string[])
            let formattedHighlights: unknown = aiHighlightTerms ?? "";
            if (typeof aiHighlightTerms === "string") {
                try {
                    const arr = JSON.parse(aiHighlightTerms);
                    if (Array.isArray(arr)) formattedHighlights = arr.join(", ");
                } catch { /* keep raw string */ }
            }

            return [{
                ...cleanInput,
                ai_decision: aiDecision ?? "",
                ai_probabilities: formattedProbs,
                ai_reasoning: aiReasoning ?? "",
                ai_highlight_terms: formattedHighlights,
                ...meta,
            }];
        }

        // If output is a JSON object (e.g. automator), spread its fields instead of adding an "output" column
        if (!hasAiOutput && r.output) {
            if (typeof r.output === "string") {
                try {
                    const parsed = JSON.parse(r.output);
                    // Array of objects (e.g. extract-data records) → one row per record
                    if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === "object") {
                        return parsed.map((rec: Record<string, unknown>) => ({ ...input, ...rec, ...meta }));
                    }
                    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                        return [{ ...input, ...parsed, ...meta }];
                    }
                } catch {
                    // not JSON — fall through
                }
            } else if (typeof r.output === "object") {
                return [{ ...input, ...(r.output as Record<string, unknown>), ...meta }];
            }
        }

        return [{ ...input, ...(hasAiOutput ? {} : { output: r.output }), ...meta }];
    };

    useEffect(() => {
        const fetchRunDetail = async () => {
            try {
                if (useBrowserDb) {
                    const data = await idbGetRun(id);
                    if (!data) throw new Error("Run not found");
                    setRun(data.run);
                    const typedResults = data.results as RunResult[];
                    setRawResults(typedResults);
                    if (data.run.runType === "codebook-generator") {
                        setResults(parseCodebookFromSystemPrompt(data.run.systemPrompt ?? ""));
                    } else {
                        setResults(typedResults.flatMap((r) => buildResultRows(r, data.run.runType)));
                    }
                } else {
                    const res = await fetch(`/api/runs/${id}`);
                    const data = await res.json();
                    if (data.error) throw new Error(data.error);
                    setRun(data.run);
                    setRawResults(data.results);
                    if (data.run.runType === "codebook-generator") {
                        setResults(parseCodebookFromSystemPrompt(data.run.systemPrompt ?? ""));
                    } else {
                        setResults(data.results.flatMap((r: RunResult) => buildResultRows(r, data.run.runType)));
                    }
                }
            } catch {
                toast.error("Failed to load run details");
            } finally {
                setIsLoading(false);
            }
        };
        fetchRunDetail();
    }, [id]);

    const handleDelete = async () => {
        setIsDeleting(true);
        try {
            if (useBrowserDb) {
                const result = await idbDeleteRun(id);
                if (!result.ok) throw new Error("Delete failed");
            } else {
                const res = await fetch(`/api/runs/${id}`, { method: "DELETE" });
                if (!res.ok) throw new Error("Delete failed");
            }
            toast.success("Run deleted");
            router.push("/history");
        } catch {
            toast.error("Failed to delete run");
            setIsDeleting(false);
        }
    };

    const startEditing = () => {
        setEditName(run?.inputFile ?? "");
        setIsEditing(true);
        setTimeout(() => editRef.current?.select(), 0);
    };

    const handleRename = async () => {
        const trimmed = editName.trim();
        if (!trimmed || !run || trimmed === run.inputFile) {
            setIsEditing(false);
            return;
        }
        try {
            if (useBrowserDb) {
                const result = await idbRenameRun(id, trimmed);
                if (!result.ok) throw new Error("Rename failed");
            } else {
                const res = await fetch(`/api/runs/${id}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ inputFile: trimmed }),
                });
                if (!res.ok) throw new Error("Rename failed");
            }
            setRun({ ...run, inputFile: trimmed });
            toast.success("Renamed");
        } catch {
            toast.error("Failed to rename");
        }
        setIsEditing(false);
    };

    const handleRestore = () => {
        if (!run || rawResults.length === 0) return;

        // Reconstruct original data rows from inputJson
        const data = rawResults.map((r: RunResult) => JSON.parse(r.inputJson ?? "{}"));

        // Build merged result rows (same shape tool pages expect)
        const mergedResults = rawResults.map((r: RunResult) => {
            const input = JSON.parse(r.inputJson ?? "{}");
            const hasAiOutput = Object.keys(input).some((k) => k.startsWith("ai_output"));

            // process-documents: keep output as-is, detect format from system prompt
            if (run.runType === "process-documents") {
                const sp = run.systemPrompt ?? "";
                const fmt = sp.includes("Return ONLY raw CSV") ? "csv"
                    : sp.includes("Return ONLY a JSON array") ? "json"
                    : sp.includes("Return Markdown") ? "md"
                    : sp.includes("Return Moodle GIFT format") ? "gift"
                    : "txt";
                return {
                    ...input,
                    output: r.output ?? "",
                    ...(fmt === "csv" ? { _all_records: r.output ?? "" } : {}),
                    _format: fmt,
                    status: r.status ?? "success",
                    latency_ms: Math.round(r.latency ?? 0),
                    ...(r.errorMessage ? { error_msg: r.errorMessage } : {}),
                };
            }

            let outputFields: Record<string, unknown> = {};
            if (!hasAiOutput && r.output) {
                // Qualitative coder uses ai_code as the output column name
                if (run.runType === "qualitative-coder") {
                    outputFields = { ai_code: r.output };
                // Model Comparison (formerly consensus-coder) saves judge_output as the output
                } else if (run.runType === "model-comparison" || run.runType === "consensus-coder") {
                    outputFields = { judge_output: r.output };
                } else if (typeof r.output === "string") {
                    try {
                        const parsed = JSON.parse(r.output);
                        if (Array.isArray(parsed)) {
                            // JSON array (e.g. extract-data records) — keep as _all_records
                            outputFields = { _all_records: r.output, _record_count: parsed.length };
                        } else if (parsed && typeof parsed === "object") {
                            outputFields = parsed;
                        } else {
                            outputFields = { ai_output: r.output };
                        }
                    } catch {
                        outputFields = { ai_output: r.output };
                    }
                } else if (typeof r.output === "object") {
                    outputFields = r.output as Record<string, unknown>;
                }
            }

            return {
                ...input,
                ...outputFields,
                status: r.status ?? "success",
                latency_ms: Math.round(r.latency ?? 0),
                ...(r.errorMessage ? { error_msg: r.errorMessage } : {}),
            };
        });

        const payload: RestorePayload = {
            runId: run.id,
            runType: run.runType,
            data,
            dataName: run.inputFile ?? "restored",
            systemPrompt: run.systemPrompt ?? "",
            results: mergedResults,
            provider: run.provider,
            model: run.model,
            temperature: run.temperature ?? 0,
        };

        setPendingRestore(payload);
        // Old runType slugs whose tools were renamed — route to the new URL.
        const slugAlias: Record<string, string> = {
          "consensus-coder": "model-comparison",
          "agent-panel": "mas-panel",
        };
        const targetSlug = slugAlias[run.runType] ?? run.runType;
        router.push(`/${targetSlug}`);
    };

    // Tools that support session restore (have row-level input data)
    const restorableTools = new Set([
        "transform", "qualitative-coder", "model-comparison", "consensus-coder",
        "automator", "abstract-screener",
        "ai-coder", "codebook-generator", "agent-panel", "generate",
        "extract-data", "process-documents",
    ]);
    const canRestore = run && rawResults.length > 0 && restorableTools.has(run.runType);

    if (isLoading) {
        return (
            <div className="flex h-[60vh] items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
            </div>
        );
    }

    if (!run) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] space-y-4">
                <AlertCircle className="h-12 w-12 text-muted-foreground" />
                <h2 className="text-xl font-semibold">Run not found</h2>
                <Button asChild variant="outline"><Link href="/history">Back to History</Link></Button>
            </div>
        );
    }

    const handleExport = () => {
        if (results.length === 0) return;
        const csv = [
            Object.keys(results[0]).join(","),
            ...results.map(row => Object.values(row).map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
        ].join("\n");
        const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `run_${run.id}_results.csv`;
        a.click();
    };

    return (
        <div className="space-y-6 pb-20">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Button asChild variant="ghost" size="icon">
                        <Link href="/history"><ArrowLeft className="h-5 w-5" /></Link>
                    </Button>
                    <div>
                        <div className="flex items-center gap-2">
                            {isEditing ? (
                                <Input
                                    ref={editRef}
                                    value={editName}
                                    onChange={(e) => setEditName(e.target.value)}
                                    onBlur={handleRename}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter") handleRename();
                                        if (e.key === "Escape") setIsEditing(false);
                                    }}
                                    className="h-8 text-xl font-bold w-[300px] border-none shadow-none focus-visible:ring-1 px-0"
                                />
                            ) : (
                                <>
                                    <h1 className="text-xl font-bold">{run.inputFile}</h1>
                                    <button
                                        onClick={startEditing}
                                        className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded hover:bg-muted"
                                        title="Rename"
                                    >
                                        <Pencil className="h-3.5 w-3.5" />
                                    </button>
                                </>
                            )}
                            <Badge variant="outline" className="capitalize">{run.runType}</Badge>
                        </div>
                        <p className="text-muted-foreground text-xs">Run ID: {run.id}</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {canRestore && (
                        <Button onClick={handleRestore} size="sm" variant="default">
                            <RotateCcw className="h-4 w-4 mr-2" /> Restore Session
                        </Button>
                    )}
                    <Button onClick={handleExport} size="sm" variant="outline">
                        <Download className="h-4 w-4 mr-2" /> Export CSV
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowDeleteDialog(true)}
                        className="border-red-300 text-red-600 hover:bg-red-50"
                    >
                        <Trash2 className="h-4 w-4 mr-2" /> Delete Run
                    </Button>
                </div>
            </div>

            <div className="grid md:grid-cols-5 gap-6">
                <Card className="md:col-span-1 min-w-0">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Run Stats</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <Label>Status</Label>
                                <div className="flex items-center gap-1 text-sm font-medium">
                                    {run.status === "completed" ? (
                                        <><CheckCircle2 className="h-3 w-3 text-green-500" /> Success</>
                                    ) : (
                                        <><Clock className="h-3 w-3 text-amber-500" /> {run.status}</>
                                    )}
                                </div>
                            </div>
                            <div className="space-y-1">
                                <Label>Model</Label>
                                <div className="text-sm font-medium">{run.model}</div>
                            </div>
                            <div className="space-y-1">
                                <Label>Total Rows</Label>
                                <div className="text-sm font-medium">{run.inputRows}</div>
                            </div>
                            <div className="space-y-1">
                                <Label>Avg Latency</Label>
                                <div className="text-sm font-medium">{(run.avgLatency / 1000).toFixed(2)}s</div>
                            </div>
                        </div>
                        <Separator />
                        <div className="space-y-3">
                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                                <Calendar className="h-3 w-3" />
                                {new Date(run.startedAt).toLocaleString()}
                            </div>
                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                                <Cpu className="h-3 w-3" />
                                Provider: {run.provider}
                            </div>
                        </div>
                        <Separator />
                        <Collapsible>
                            <CollapsibleTrigger className="flex items-center gap-2 w-full text-xs font-medium hover:text-foreground text-muted-foreground">
                                <ChevronRight className="h-3.5 w-3.5 transition-transform [[data-state=open]_&]:rotate-90" />
                                System Prompt Used
                            </CollapsibleTrigger>
                            <CollapsibleContent>
                                <pre className="text-xs font-mono bg-muted/10 p-3 mt-2 rounded border whitespace-pre-wrap break-words">
                                    {run.systemPrompt || "—"}
                                </pre>
                            </CollapsibleContent>
                        </Collapsible>
                    </CardContent>
                </Card>

                <div className="md:col-span-4 min-w-0">
                    {(() => {
                        const sp = run.systemPrompt ?? "";

                        // ── Helpers ──────────────────────────────────────────────
                        const detectFormat = () => {
                            if (sp.includes("Return ONLY raw CSV")) return { label: "CSV", ext: "csv", font: "font-mono" } as const;
                            if (sp.includes("Return ONLY a JSON array") || sp.includes("Format: json")) return { label: "JSON", ext: "json", font: "font-mono" } as const;
                            if (sp.includes("Return Markdown") || sp.includes("Format: Markdown")) return { label: "Markdown", ext: "md", font: "font-sans" } as const;
                            if (sp.includes("Return Moodle GIFT format") || sp.includes("Format: Moodle GIFT")) return { label: "GIFT", ext: "gift", font: "font-sans" } as const;
                            if (sp.includes("Format: plain readable text") || sp.includes("Return plain readable text")) return { label: "Free Text", ext: "txt", font: "font-sans" } as const;
                            return null;
                        };

                        const fmt = detectFormat();

                        const copyToClipboard = (text: string, idx: number) => {
                            navigator.clipboard.writeText(text).then(() => {
                                setCopiedIdx(idx);
                                setTimeout(() => setCopiedIdx(null), 2000);
                            });
                        };

                        const parseCsvToRows = (csvText: string): Record<string, unknown>[] => {
                            const raw = csvText.replace(/^```(?:csv|json)?\s*/im, "").replace(/\s*```\s*$/m, "").trim();
                            const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
                            if (lines.length < 2) return [];
                            const parseCsvRow = (line: string): string[] => {
                                const values: string[] = [];
                                let current = "";
                                let inQuotes = false;
                                for (let i = 0; i < line.length; i++) {
                                    const ch = line[i];
                                    if (ch === '"') {
                                        if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
                                        else { inQuotes = !inQuotes; }
                                    } else if (ch === "," && !inQuotes) {
                                        values.push(current); current = "";
                                    } else {
                                        current += ch;
                                    }
                                }
                                values.push(current);
                                return values.map((v) => v.trim());
                            };
                            const headers = parseCsvRow(lines[0]);
                            const rows: Record<string, unknown>[] = [];
                            for (let li = 1; li < lines.length; li++) {
                                const values = parseCsvRow(lines[li]);
                                const row: Record<string, unknown> = {};
                                headers.forEach((h, i) => { if (h) row[h] = values[i] ?? ""; });
                                rows.push(row);
                            }
                            return rows;
                        };

                        // ── Freetext-like runs (generate or process-documents non-CSV) ──
                        const isFreetextGenerate = run.runType === "generate" && fmt && fmt.ext !== "csv" && fmt.ext !== "json" &&
                            rawResults.length === 1 && typeof rawResults[0].output === "string" && rawResults[0].output.length > 0;

                        const isJsonGenerate = run.runType === "generate" && rawResults.length > 0 && fmt?.ext === "json";

                        const isProcessDocs = run.runType === "process-documents";
                        const isProcessDocsCsv = isProcessDocs && fmt?.ext === "csv";
                        const isProcessDocsText = isProcessDocs && !isProcessDocsCsv;

                        // ── Process-documents CSV → parse into proper table columns ──
                        if (isProcessDocsCsv && rawResults.length > 0) {
                            const tableRows: Record<string, unknown>[] = [];
                            for (const r of rawResults) {
                                if (r.status !== "success" || !r.output) continue;
                                const input = JSON.parse(r.inputJson ?? "{}");
                                const docName = (input.document_name as string) ?? "Document";
                                const parsed = parseCsvToRows(r.output as string);
                                for (const row of parsed) {
                                    tableRows.push({ document_name: docName, ...row });
                                }
                            }
                            if (tableRows.length > 0) {
                                return (
                                    <div>
                                        <div className="px-4 py-2.5 border border-b-0 rounded-t-lg bg-muted/20 text-sm font-medium flex items-center justify-between flex-wrap gap-2">
                                            <span>Processed Documents — {rawResults.filter((r) => r.status === "success").length} file{rawResults.filter((r) => r.status === "success").length !== 1 ? "s" : ""} — {tableRows.length} rows</span>
                                            <ExportDropdown data={tableRows} filename="run_results" />
                                        </div>
                                        <div className="border rounded-b-lg">
                                            <DataTable data={tableRows} />
                                        </div>
                                    </div>
                                );
                            }
                        }

                        // ── Freetext output (generate or process-documents non-CSV) ──
                        if ((isFreetextGenerate || isJsonGenerate || isProcessDocsText) && rawResults.length > 0 && fmt) {
                            // Build per-entry blocks from raw results
                            const entries = rawResults
                                .filter((r) => r.status === "success" && r.output)
                                .map((r) => {
                                    const input = JSON.parse(r.inputJson ?? "{}");
                                    const name = isProcessDocs
                                        ? ((input.document_name as string) ?? "Document")
                                        : ((input.description as string) ?? "Generated");
                                    return { name, output: r.output as string };
                                });

                            // For JSON generate, reconstruct array with per-row status + latency_ms
                            let combinedRaw: string;
                            if (isJsonGenerate) {
                                const rows = rawResults
                                    .filter((r) => r.status === "success" && r.output)
                                    .map((r) => {
                                        let parsed: unknown;
                                        try { parsed = JSON.parse(r.output as string); } catch { parsed = r.output; }
                                        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                                            return {
                                                ...(parsed as Record<string, unknown>),
                                                status: r.status ?? "success",
                                                latency_ms: Math.round(r.latency ?? 0),
                                            };
                                        }
                                        return parsed;
                                    });
                                combinedRaw = JSON.stringify(rows, null, 2);
                            } else if (entries.length === 1) {
                                combinedRaw = entries[0].output;
                            } else {
                                combinedRaw = entries.map((e) => `=== ${e.name} ===\n\n${e.output}`).join("\n\n---\n\n");
                            }

                            const headerLabel = isProcessDocs
                                ? `Processed Documents — ${entries.length} file${entries.length !== 1 ? "s" : ""} — ${fmt.label}`
                                : `Generated Output — ${fmt.label}`;

                            return (
                                <div className="border rounded-lg overflow-hidden">
                                    <div className="px-4 py-2.5 border-b bg-muted/20 text-sm font-medium flex items-center justify-between flex-wrap gap-2">
                                        <span>{headerLabel}</span>
                                        <div className="flex items-center gap-2">
                                            <button
                                                onClick={() => copyToClipboard(combinedRaw, -1)}
                                                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                                            >
                                                {copiedIdx === -1 ? (
                                                    <><Check className="h-3 w-3 text-green-600" /> Copied</>
                                                ) : (
                                                    <><Copy className="h-3 w-3" /> Copy</>
                                                )}
                                            </button>
                                            <Button variant="outline" size="sm" className="text-xs" onClick={() => {
                                                const blob = new Blob([combinedRaw], { type: "text/plain;charset=utf-8;" });
                                                const url = URL.createObjectURL(blob);
                                                const a = document.createElement("a"); a.href = url; a.download = `${isProcessDocs ? "processed_documents" : `generated_${run.id}`}.${fmt.ext}`; a.click();
                                                URL.revokeObjectURL(url);
                                            }}>
                                                <Download className="h-3.5 w-3.5 mr-1.5" /> Download .{fmt.ext}
                                            </Button>
                                        </div>
                                    </div>
                                    <pre className={`p-4 text-sm whitespace-pre-wrap bg-muted/10 leading-relaxed ${fmt.font}`}>
                                        {combinedRaw}
                                    </pre>
                                </div>
                            );
                        }

                        return (
                            <div>
                                <div className="px-4 py-2.5 border border-b-0 rounded-t-lg bg-muted/20 text-sm font-medium flex items-center justify-between flex-wrap gap-2">
                                    <span>Processed Results — {results.length} rows</span>
                                    <ExportDropdown data={results} filename="run_results" />
                                </div>
                                <div className="border rounded-b-lg">
                                    <DataTable data={results} />
                                </div>
                            </div>
                        );
                    })()}
                </div>
            </div>

            <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Run?</DialogTitle>
                        <DialogDescription>
                            This will permanently delete this run and all its results. This cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setShowDeleteDialog(false)} disabled={isDeleting}>
                            Cancel
                        </Button>
                        <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
                            {isDeleting ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Trash2 className="h-4 w-4 mr-2" />}
                            Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}

function Label({ children }: { children: React.ReactNode }) {
    return <div className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">{children}</div>;
}

function Separator() {
    return <div className="h-px bg-muted w-full my-1" />;
}
