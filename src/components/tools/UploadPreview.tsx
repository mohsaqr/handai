"use client";

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DataTable, ExportDropdown } from "./DataTable";
import { SAMPLE_DATASETS } from "@/lib/sample-data";
import { parseStructuredFile, getFileExt } from "@/lib/parse-file";
import { toast } from "sonner";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Row = Record<string, any>;

interface UploadPreviewProps {
  data: Row[];
  dataName: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onDataLoaded: (data: any[], name: string) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSampleLoad?: (key: string, data: any[], name: string) => void;
  maxPreviewRows?: number;
  customSamplePicker?: React.ReactNode;
  // Kept for backwards-compatibility with existing call sites; the sample
  // picker is now always pinned to the right of the drop area (see below).
  samplePickerPosition?: "above" | "below";
  accept?: Record<string, string[]>;
  children?: React.ReactNode;
  bannerExtra?: React.ReactNode;
}

export function UploadPreview({
  data,
  dataName,
  onDataLoaded,
  onSampleLoad,
  maxPreviewRows: _maxPreviewRows = 5, // eslint-disable-line @typescript-eslint/no-unused-vars
  customSamplePicker,
  samplePickerPosition: _samplePickerPosition, // eslint-disable-line @typescript-eslint/no-unused-vars
  accept,
  children,
  bannerExtra,
}: UploadPreviewProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLoadSample = (key: string) => {
    const s = SAMPLE_DATASETS[key];
    if (!s) return;
    if (onSampleLoad) {
      onSampleLoad(key, s.data as Row[], s.name);
    } else {
      onDataLoaded(s.data as Row[], s.name);
      toast.success(`Loaded ${s.data.length} rows from ${s.name}`);
    }
  };

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setIsProcessing(true);
      setError(null);
      try {
        const rows = await parseStructuredFile(file);
        if (!rows) {
          const ext = getFileExt(file.name);
          setError(ext === "ris" ? "No valid records found in RIS file" : `Failed to parse .${ext} file`);
          setIsProcessing(false);
          return;
        }
        onDataLoaded(rows, file.name);
      } catch (err: unknown) {
        setError(`Failed to process file: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [onDataLoaded],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept || {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
      "application/json": [".json"],
      "application/x-research-info-systems": [".ris"],
      "text/plain": [".ris"],
    },
    multiple: false,
  });

  // Default "Load sample…" dropdown — matches the SmartFileUpload look so every
  // tool's upload area is identical. Call sites may still pass their own.
  const sampleEl = customSamplePicker ?? (
    <Select value="" onValueChange={handleLoadSample}>
      <SelectTrigger className="w-[200px] h-9 text-xs shrink-0">
        <SelectValue placeholder="-- Load sample..." />
      </SelectTrigger>
      <SelectContent>
        {Object.keys(SAMPLE_DATASETS).map((key) => (
          <SelectItem key={key} value={key} className="text-xs">
            {SAMPLE_DATASETS[key].name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  return (
    <div className="space-y-4">
      <div className="relative flex w-full items-center justify-center">
        <div
          {...getRootProps()}
          className={`w-[42rem] max-w-full border-2 border-dashed rounded-xl px-6 py-10 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-primary bg-primary/10"
              : "border-primary/40 bg-primary/[0.03] hover:border-primary/70 hover:bg-primary/5"
          } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
        >
          <input {...getInputProps()} />
          <div className="flex items-center justify-center gap-2.5">
            <Upload className="h-6 w-6 text-muted-foreground shrink-0" />
            <span className="text-base font-medium">
              {isDragActive ? "Drop the file here" : "Click or drag file to upload"}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-muted-foreground">CSV, Excel, JSON, RIS</p>
          {isProcessing && <div className="mt-2 text-sm animate-pulse text-primary">Processing…</div>}
        </div>

        <div className="absolute right-0 top-1/2 -translate-y-1/2">{sampleEl}</div>
      </div>

      {error && (
        <div className="flex items-center justify-center text-sm text-destructive gap-2">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {children}

      {data.length > 0 && (
        <div className="border rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 border-b bg-muted/20 text-sm font-medium flex items-center justify-between flex-wrap gap-2">
            <span className="flex items-center">
              Data Preview — {data.length} rows{dataName ? ` · ${dataName}` : ""}
              {bannerExtra}
            </span>
            <ExportDropdown data={data} filename="preview" />
          </div>
          <DataTable data={data} />
        </div>
      )}
    </div>
  );
}
