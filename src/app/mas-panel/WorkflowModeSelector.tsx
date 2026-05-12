"use client";

import { Wand2, GitBranch, Users } from "lucide-react";
import { type WorkflowMode } from "./workflow-types";

interface Option {
  id: WorkflowMode;
  title: string;
  subtitle: string;
  description: string;
  Icon: React.ComponentType<{ className?: string }>;
}

const OPTIONS: Option[] = [
  {
    id: "reconcilier",
    title: "Reconcilier",
    subtitle: "Hierarchical — reconciler on top, workers below",
    description:
      "Multiple workers analyze the task in parallel. The topmost card reconciles their outputs into one final answer.",
    Icon: Wand2,
  },
  {
    id: "sequential",
    title: "Sequential",
    subtitle: "Pipeline — output of step N feeds step N+1",
    description:
      "Each step processes the previous step's output. Use when later agents build on earlier work.",
    Icon: GitBranch,
  },
  {
    id: "deliberation",
    title: "Deliberation",
    subtitle: "Peer network — all agents talk to each other",
    description:
      "All agents deliberate over multiple rounds, seeing each other's outputs. Converges on a shared answer.",
    Icon: Users,
  },
];

interface Props {
  value: WorkflowMode | null;
  onChange: (mode: WorkflowMode) => void;
}

export function WorkflowModeSelector({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
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
