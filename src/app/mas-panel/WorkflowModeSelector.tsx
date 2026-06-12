"use client";

import { Gavel, ArrowRight, MessagesSquare, Workflow } from "lucide-react";
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
    title: "Judge",
    subtitle: "Workers answer in parallel, one judge decides",
    description:
      "Every worker tackles the same task on its own, then a single judge agent reviews all their answers and produces the final result. Best for a vetted, consensus answer.",
    Icon: Gavel,
  },
  {
    id: "personalized",
    title: "Individual / Personalized",
    subtitle: "Build it yourself",
    description:
      "Place agents freely and draw your own connections. Each agent feeds the ones you link it to. Best when your process doesn't fit the other templates.",
    Icon: Workflow,
  },
  {
    id: "sequential",
    title: "Sequential",
    subtitle: "Assembly line, each agent builds on the last",
    description:
      "Agents run one after another — each receives the previous agent's output as its input. Best when every step refines or extends the work before it.",
    Icon: ArrowRight,
  },
  {
    id: "deliberation",
    title: "Deliberation",
    subtitle: "Group discussion over several rounds",
    description:
      "All agents see each other's answers and revise across multiple rounds until they converge on a shared answer. Best for hard questions that benefit from back-and-forth.",
    Icon: MessagesSquare,
  },
];

interface Props {
  value: WorkflowMode | null;
  onChange: (mode: WorkflowMode) => void;
}

export function WorkflowModeSelector({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
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
