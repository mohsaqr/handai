"use client";

import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { User, X } from "lucide-react";
import { type Agent, avatarStyle } from "@/lib/agent-library";
import { type WorkflowStep } from "./workflow-types";

export type StepStatus = "pending" | "running" | "done" | "error";

interface Props {
  step: WorkflowStep;
  index: number;
  label?: string;                 // e.g. "Reconciler" / "Worker 2" / "Step 1"
  showIndex?: boolean;            // Hide the numbered circle badge (e.g. for reconcilier)
  taskPlaceholder?: string;       // Override the Describe Task field placeholder
  samplePrompts?: Record<string, string>;  // Optional preset picker that fills taskDescription
  status?: StepStatus;            // Coloured ring during batch processing
  compact?: boolean;              // Tighter padding + smaller avatar — used by Sequential layout
  agents: Agent[];
  onUpdate: (step: WorkflowStep) => void;
  onRemove: () => void;
  canRemove: boolean;
}

export function WorkflowStepCard({
  step,
  index,
  label,
  showIndex = true,
  taskPlaceholder = "What should this agent do in this step?",
  samplePrompts,
  status,
  compact = false,
  agents,
  onUpdate,
  onRemove,
  canRemove,
}: Props) {
  const statusRing =
    status === "running" ? "ring-2 ring-red-400 animate-pulse" :
    status === "done"    ? "ring-2 ring-green-400" :
    status === "error"   ? "ring-2 ring-amber-400" :
    "";

  const assignedAgent = agents.find((a) => a.id === step.agentId);
  const avatar = assignedAgent?.avatar;

  return (
    <div className={`border rounded-lg ${compact ? "p-2 gap-3" : "p-3 gap-5"} bg-background relative transition-shadow flex ${statusRing}`}>
      {canRemove && (
        <button
          onClick={onRemove}
          className="absolute top-2 right-2 h-7 w-7 rounded-md border bg-background flex items-center justify-center text-muted-foreground hover:text-destructive hover:border-destructive transition-colors shadow-sm"
          title="Remove step"
        >
          <X className="h-[18px] w-[18px]" strokeWidth={2.5} />
        </button>
      )}

      <div
        className={`shrink-0 ${compact ? "w-20 h-20" : "w-24 h-24"} rounded-lg overflow-hidden flex items-center justify-center bg-muted/40 ${
          typeof avatar === "number" ? "border" : "border border-dashed border-muted-foreground/40"
        }`}
      >
        {typeof avatar === "number" ? (
          <div className="w-full h-full" style={avatarStyle(avatar)} aria-hidden />
        ) : (
          <User className={compact ? "h-9 w-9 text-muted-foreground/60" : "h-10 w-10 text-muted-foreground/60"} />
        )}
      </div>

      <div className={`flex-1 min-w-0 ${compact ? "space-y-1" : "space-y-2"}`}>
        <div className="flex items-center gap-2 pr-6">
          {showIndex && (
            <div className="h-6 w-6 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0">
              {index + 1}
            </div>
          )}
          <div className="text-sm font-semibold truncate">
            {label ?? `Step ${index + 1}`}
          </div>
        </div>

      <div className="space-y-1">
        <Label className="text-[10px] text-muted-foreground">Agent</Label>
        <Select
          value={step.agentId ?? ""}
          onValueChange={(v) => onUpdate({ ...step, agentId: v || null })}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="Pick an agent…" />
          </SelectTrigger>
          <SelectContent>
            {agents.length === 0 && (
              <SelectItem value="__none" disabled className="text-xs">
                No agents configured
              </SelectItem>
            )}
            {agents.map((a) => (
              <SelectItem key={a.id} value={a.id} className="text-xs">
                {a.name || "(unnamed)"} — {a.model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <Label className="text-[10px] text-muted-foreground">Describe Task</Label>
          {samplePrompts && (
            <Select
              onValueChange={(key) => {
                const preset = samplePrompts[key];
                if (preset) onUpdate({ ...step, taskDescription: preset });
              }}
            >
              <SelectTrigger className="h-6 w-[140px] text-[10px]">
                <SelectValue placeholder="Load sample…" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(samplePrompts).map((key) => (
                  <SelectItem key={key} value={key} className="text-xs">
                    {key}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <Textarea
          value={step.taskDescription}
          onChange={(e) => onUpdate({ ...step, taskDescription: e.target.value })}
          placeholder={taskPlaceholder}
          rows={compact ? 1 : 2}
          className={`text-xs resize-y ${compact ? "min-h-[28px]" : "min-h-[48px]"}`}
        />
      </div>

      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground py-0.5">
          Persona + knowledge overrides
        </summary>
        <div className="space-y-2 pt-2">
          <div className="space-y-1">
            <Label className="text-[10px] text-muted-foreground">Persona (adds to agent&apos;s own)</Label>
            <Textarea
              value={step.persona}
              onChange={(e) => onUpdate({ ...step, persona: e.target.value })}
              placeholder="Step-specific persona…"
              rows={2}
              className="text-xs resize-y min-h-[40px]"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-[10px] text-muted-foreground">Additional Knowledge (adds to agent&apos;s own)</Label>
            <Textarea
              value={step.additionalKnowledge}
              onChange={(e) => onUpdate({ ...step, additionalKnowledge: e.target.value })}
              placeholder="Reference material for this step…"
              rows={2}
              className="text-xs resize-y min-h-[40px]"
            />
          </div>
        </div>
      </details>
      </div>
    </div>
  );
}

/** Compact placeholder shown when a slot is empty */
export function EmptySlot({ onAdd }: { onAdd: () => void }) {
  return (
    <Button
      variant="outline"
      className="border-dashed w-full h-full min-h-[120px] text-xs text-muted-foreground"
      onClick={onAdd}
    >
      + Add step
    </Button>
  );
}
