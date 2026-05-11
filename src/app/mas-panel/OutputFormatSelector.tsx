"use client";

import { Rows3, FileText } from "lucide-react";

export type OutputFormat = "per-row" | "document";

interface Option {
  id: OutputFormat;
  title: string;
  subtitle: string;
  description: string;
  Icon: React.ComponentType<{ className?: string }>;
}

const OPTIONS: Option[] = [
  {
    id: "per-row",
    title: "One output per row",
    subtitle: "Workflow runs once per row",
    description:
      "Each row of the uploaded file is processed independently. The result table has one row per input row.",
    Icon: Rows3,
  },
  {
    id: "document",
    title: "One output for the whole file",
    subtitle: "Workflow sees every row at once",
    description:
      "The entire file is sent as a single document (JSON of all rows). Produces a single result row — useful for tasks like picking the best candidate from a catalogue.",
    Icon: FileText,
  },
];

interface Props {
  value: OutputFormat;
  onChange: (v: OutputFormat) => void;
}

export function OutputFormatSelector({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {OPTIONS.map((opt) => {
        const selected = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            className={`text-left rounded-lg border-2 p-4 transition-colors ${
              selected
                ? "border-primary bg-primary/5 shadow-sm"
                : "border-border hover:border-muted-foreground/30 hover:bg-muted/30"
            }`}
          >
            <div className="flex items-center gap-2 mb-1.5">
              <opt.Icon className={`h-5 w-5 ${selected ? "text-primary" : "text-muted-foreground"}`} />
              <div className="text-base font-semibold">{opt.title}</div>
            </div>
            <div className="text-xs text-muted-foreground font-medium mb-2">{opt.subtitle}</div>
            <div className="text-xs text-muted-foreground leading-relaxed">{opt.description}</div>
          </button>
        );
      })}
    </div>
  );
}
