"use client";

import { type ReactNode } from "react";
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
import { User, X, Plus, FileText } from "lucide-react";
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
  /** Personalized & Sequential — render the editable Input DATA chip row. */
  showInputData?: boolean;
  /** Page-level selected columns available to this card. */
  inputCols?: string[];
  /** Personalized — connected upstream sources feeding this card. */
  connectedSources?: { id: string; label: string }[];
  /** Sequential — label of the previous step (e.g. "Step 1"), or null. */
  prevStepLabel?: string | null;
  /** Non-removable info chips for intrinsic inputs (e.g. the manager's
   *  "Workers' outputs", which it always receives and can't opt out of). */
  staticSources?: string[];
  /** Name of an uploaded unstructured file (PDF/DOCX/TXT). When set, a card that
   *  relies on the original input shows a document chip instead of "no input",
   *  since the extracted file text IS what such a card receives. */
  documentInput?: string;
}

function Chip({
  text,
  onRemove,
  icon,
}: {
  text: string;
  onRemove: () => void;
  icon?: ReactNode;
}) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
      {icon}
      <span className="font-mono truncate max-w-[120px]">{text}</span>
      <button
        type="button"
        onClick={onRemove}
        className="hover:text-destructive shrink-0"
        title="Remove from this card's input"
      >
        <X className="h-3 w-3" strokeWidth={2.5} />
      </button>
    </span>
  );
}

// Dashed "+ label" pill shown when an opt-out-able input (previous-step output,
// document) has been removed, letting the user add it back.
function RestoreChip({ label, onClick, title }: { label: string; onClick: () => void; title?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="inline-flex items-center gap-1 rounded-md border border-dashed px-1.5 py-0.5 text-[10px] text-muted-foreground hover:text-foreground"
    >
      <Plus className="h-3 w-3" /> {label}
    </button>
  );
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
  showInputData = false,
  inputCols = [],
  connectedSources = [],
  prevStepLabel = null,
  staticSources = [],
  documentInput,
}: Props) {
  const excluded = new Set(step.excludedCols ?? []);
  const keptCols = inputCols.filter((c) => !excluded.has(c));
  const removedCols = inputCols.filter((c) => excluded.has(c));
  const removeCol = (c: string) =>
    onUpdate({ ...step, excludedCols: [...(step.excludedCols ?? []), c] });
  const restoreCol = (c: string) =>
    onUpdate({ ...step, excludedCols: (step.excludedCols ?? []).filter((x) => x !== c) });
  const removeSource = (srcId: string) =>
    onUpdate({ ...step, inputs: (step.inputs ?? []).filter((s) => s !== srcId) });

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
        className={`shrink-0 ${compact ? "w-28 h-28" : "w-32 h-32"} rounded-lg overflow-hidden flex items-center justify-center bg-muted/40 ${
          typeof avatar === "number" ? "border" : "border border-dashed border-muted-foreground/40"
        }`}
      >
        {typeof avatar === "number" ? (
          <div className="w-full h-full" style={avatarStyle(avatar)} aria-hidden />
        ) : (
          <User className={compact ? "h-12 w-12 text-muted-foreground/60" : "h-14 w-14 text-muted-foreground/60"} />
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
          onValueChange={(v) => {
            // Picking an agent seeds the Main Prompt box with that agent's main
            // system prompt (goal). Only seed when the box is empty or still
            // holds the previous agent's goal (an untouched seed) — never clobber
            // a prompt the user has edited.
            const picked = v ? agents.find((a) => a.id === v) : null;
            const prevGoal = (step.agentId
              ? agents.find((a) => a.id === step.agentId)?.goal
              : ""
            )?.trim() ?? "";
            const current = step.taskDescription.trim();
            const newGoal = picked?.goal?.trim() ?? "";
            const seed = newGoal && (current === "" || current === prevGoal);
            onUpdate({
              ...step,
              agentId: v || null,
              ...(seed ? { taskDescription: newGoal } : {}),
            });
          }}
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

      {showInputData && (
        <div className="space-y-1">
          <Label className="text-[10px] text-muted-foreground">Input DATA</Label>
          <div className="flex flex-wrap gap-1 items-center">
            {keptCols.map((c) => (
              <Chip key={`col-${c}`} text={c} onRemove={() => removeCol(c)} />
            ))}
            {documentInput && !step.ignoreDocument && (
              <Chip
                text={documentInput}
                icon={<FileText className="h-3 w-3 shrink-0" />}
                onRemove={() => onUpdate({ ...step, ignoreDocument: true })}
              />
            )}
            {connectedSources.map((s) => (
              <Chip
                key={`src-${s.id}`}
                text={`${s.label} output`}
                onRemove={() => removeSource(s.id)}
              />
            ))}
            {staticSources.map((label) => (
              <span
                key={`static-${label}`}
                className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                title="Always provided to this card"
              >
                <span className="font-mono truncate max-w-[140px]">{label}</span>
              </span>
            ))}
            {prevStepLabel && !step.ignorePrevOutput && (
              <Chip
                text={`${prevStepLabel} output`}
                onRemove={() => onUpdate({ ...step, ignorePrevOutput: true })}
              />
            )}
            {prevStepLabel && step.ignorePrevOutput && (
              <RestoreChip
                label={`${prevStepLabel} output`}
                onClick={() => onUpdate({ ...step, ignorePrevOutput: false })}
              />
            )}
            {documentInput && step.ignoreDocument && (
              <RestoreChip
                label={documentInput}
                onClick={() => onUpdate({ ...step, ignoreDocument: false })}
                title="Send the uploaded document's text to this card"
              />
            )}
            {removedCols.length > 0 && (
              <details className="relative inline-block text-[10px]">
                <summary className="list-none cursor-pointer inline-flex items-center gap-1 rounded-md border border-dashed px-1.5 py-0.5 text-muted-foreground hover:text-foreground">
                  <Plus className="h-3 w-3" /> column
                </summary>
                <div className="absolute left-0 z-30 mt-1 max-h-48 min-w-[140px] overflow-y-auto rounded-md border bg-background p-1 shadow-md">
                  {removedCols.map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={(e) => {
                        restoreCol(c);
                        e.currentTarget.closest("details")?.removeAttribute("open");
                      }}
                      className="block w-full truncate rounded px-2 py-1 text-left font-mono hover:bg-muted"
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </details>
            )}
            {keptCols.length === 0 &&
              connectedSources.length === 0 &&
              staticSources.length === 0 &&
              !(prevStepLabel && !step.ignorePrevOutput) &&
              !documentInput && (
                <span className="text-[10px] text-muted-foreground italic">
                  no input — uses original input
                </span>
              )}
          </div>
        </div>
      )}

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <Label className="text-[10px] text-muted-foreground">Main Prompt</Label>
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
          rows={2}
          className={`text-xs resize-y ${compact ? "min-h-[56px]" : "min-h-[48px]"}`}
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
