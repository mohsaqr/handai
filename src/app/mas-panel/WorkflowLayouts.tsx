"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { type Agent } from "@/lib/agent-library";
import { type WorkflowMode, type WorkflowStep } from "./workflow-types";
import { WorkflowStepCard, type StepStatus } from "./WorkflowStepCard";
import { SAMPLE_RECONCILER_PROMPTS } from "./reconciler-samples";

interface LayoutProps {
  mode: WorkflowMode;
  steps: WorkflowStep[];
  agents: Agent[];
  stepStatuses?: Record<string, StepStatus>;
  onUpdate: (id: string, step: WorkflowStep) => void;
  onRemove: (id: string) => void;
  onAdd: () => void;
}

export function WorkflowLayout(props: LayoutProps) {
  if (props.mode === "sequential") return <SequentialSLayout {...props} />;
  if (props.mode === "reconcilier") return <ReconcilierHierarchyLayout {...props} />;
  return <DeliberationNetworkLayout {...props} />;
}

function statusFor(stepId: string, statuses?: Record<string, StepStatus>): StepStatus | undefined {
  return statuses?.[stepId];
}

// ── Shared connector components (dashed lines) ───────────────────────────────
// Structure mirrors ai-agents/page.tsx:172 so arrows line up with card columns.

function SConnector({ direction }: {
  direction: "left-to-right" | "right-to-left" | "down-left" | "down-right";
}) {
  const color = "text-muted-foreground";

  if (direction === "left-to-right") {
    return (
      <div className="flex flex-col items-center justify-center w-40 self-center">
        <svg width="160" height="28" viewBox="0 0 160 28" fill="none" className={color}>
          <circle cx="5" cy="14" r="4" fill="currentColor" />
          <path d="M9 14 H140" stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points="140,5 158,14 140,23" fill="currentColor" />
        </svg>
      </div>
    );
  }
  if (direction === "right-to-left") {
    return (
      <div className="flex flex-col items-center justify-center w-40 self-center">
        <svg width="160" height="28" viewBox="0 0 160 28" fill="none" className={color}>
          <circle cx="155" cy="14" r="4" fill="currentColor" />
          <path d="M151 14 H20" stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points="20,5 2,14 20,23" fill="currentColor" />
        </svg>
      </div>
    );
  }

  // down-left / down-right — vertical arrow positioned under a card column
  const alignRight = direction === "down-right";
  const arrow = (
    <div className="flex-1 min-w-0 flex flex-col items-center py-2">
      <svg width="28" height="56" viewBox="0 0 28 56" fill="none" className={color}>
        <circle cx="14" cy="4" r="4" fill="currentColor" />
        <path d="M14 8 V44" stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
        <polygon points="6,44 14,56 22,44" fill="currentColor" />
      </svg>
    </div>
  );
  const spacer = <div className="flex-1 min-w-0" />;
  const gap = <div className="w-40 shrink-0" />;

  return (
    <div className="flex gap-0">
      {alignRight ? spacer : arrow}
      {gap}
      {alignRight ? arrow : spacer}
    </div>
  );
}

// ── Sequential S-layout (2 per row, U-turn between rows) ─────────────────────

function SequentialSLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd }: LayoutProps) {
  const rows: WorkflowStep[][] = [];
  for (let i = 0; i < steps.length; i += 2) rows.push(steps.slice(i, i + 2));

  return (
    <div className="space-y-0">
      {rows.map((row, rowIdx) => {
        const isEvenRow = rowIdx % 2 === 0;
        const orderedRow = isEvenRow ? row : [...row].reverse();
        const globalBase = rowIdx * 2;

        return (
          <div key={rowIdx}>
            {/* Row of cards */}
            <div className="flex gap-0 items-stretch">
              {/* Odd row with a single card: spacer on LEFT so card sits on the right */}
              {row.length === 1 && !isEvenRow && (
                <>
                  <div className="flex-1 min-w-0" />
                  <div className="w-40 shrink-0" />
                </>
              )}
              {orderedRow.map((step, colIdx) => {
                const globalIdx = isEvenRow
                  ? globalBase + colIdx
                  : globalBase + (row.length - 1 - colIdx);
                const isLastInRow = colIdx === orderedRow.length - 1;
                return (
                  <React.Fragment key={step.id}>
                    <div className="flex-1 min-w-0">
                      <WorkflowStepCard
                        step={step}
                        index={globalIdx}
                        label={`Step ${globalIdx + 1}`}
                        status={statusFor(step.id, stepStatuses)}
                        agents={agents}
                        compact
                        onUpdate={(s) => onUpdate(step.id, s)}
                        onRemove={() => onRemove(step.id)}
                        canRemove={steps.length > 2}
                      />
                    </div>
                    {!isLastInRow && row.length > 1 && (
                      <SConnector direction={isEvenRow ? "left-to-right" : "right-to-left"} />
                    )}
                  </React.Fragment>
                );
              })}
              {/* Even row with a single card: spacer on RIGHT so card sits on the left */}
              {row.length === 1 && isEvenRow && (
                <>
                  <div className="w-40 shrink-0" />
                  <div className="flex-1 min-w-0" />
                </>
              )}
            </div>

            {/* U-turn connector between rows — lands under the card where the S turns */}
            {rowIdx < rows.length - 1 && (() => {
              const nextRow = rows[rowIdx + 1];
              let dir: "down-right" | "down-left";
              if (nextRow.length === 1) {
                // Single card in next row: odd row → right, even row → left
                dir = (rowIdx + 1) % 2 !== 0 ? "down-right" : "down-left";
              } else {
                dir = isEvenRow ? "down-right" : "down-left";
              }
              return <SConnector direction={dir} />;
            })()}
          </div>
        );
      })}

      <div className="pt-4">
        <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={onAdd}>
          <Plus className="h-3.5 w-3.5" /> Add step
        </Button>
      </div>
    </div>
  );
}

// ── Reconcilier hierarchy: top card = reconciler, workers below with lines up ─

function ReconcilierHierarchyLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd }: LayoutProps) {
  const reconciler = steps[0];
  const workers = steps.slice(1);

  return (
    <div className="space-y-0">
      {/* Reconciler row — centered, wider than workers */}
      {reconciler && (
        <div className="flex justify-center">
          <div className="w-full max-w-3xl">
            <WorkflowStepCard
              step={reconciler}
              index={0}
              label="Reconciler (top of tree)"
              showIndex={false}
              taskPlaceholder="How should this agent reconcile the workers' outputs into a final answer?"
              samplePrompts={SAMPLE_RECONCILER_PROMPTS}
              status={statusFor(reconciler.id, stepStatuses)}
              agents={agents}
              onUpdate={(s) => onUpdate(reconciler.id, s)}
              onRemove={() => onRemove(reconciler.id)}
              canRemove={steps.length > 2}
            />
          </div>
        </div>
      )}

      {/* Tree spokes — thick dashed arrows pointing UP from each worker to the reconciler */}
      {workers.length > 0 && (
        <div className="relative h-28" aria-hidden>
          <svg
            className="absolute inset-0 w-full h-full text-muted-foreground"
            preserveAspectRatio="none"
            style={{ overflow: "visible" }}
          >
            <defs>
              <marker
                id="reconciler-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
              </marker>
            </defs>
            {workers.map((_, i) => {
              // Evenly-spaced X from 0..100% for N workers
              const xPct = workers.length === 1 ? 50 : (i / (workers.length - 1)) * 80 + 10;
              return (
                <line
                  key={i}
                  x1={`${xPct}%`}
                  y1="100%"
                  x2="50%"
                  y2="0"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeDasharray="6 4"
                  markerEnd="url(#reconciler-arrow)"
                />
              );
            })}
          </svg>
        </div>
      )}

      {/* Worker cards row */}
      {workers.length > 0 && (
        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(workers.length, 3)}, minmax(0, 1fr))` }}>
          {workers.map((w, i) => (
            <WorkflowStepCard
              key={w.id}
              step={w}
              index={i + 1}
              label={`Worker ${i + 1}`}
              showIndex={false}
              taskPlaceholder="What angle or analysis should this worker contribute?"
              status={statusFor(w.id, stepStatuses)}
              agents={agents}
              onUpdate={(s) => onUpdate(w.id, s)}
              onRemove={() => onRemove(w.id)}
              canRemove={steps.length > 2}
            />
          ))}
        </div>
      )}

      <div className="pt-4">
        <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={onAdd}>
          <Plus className="h-3.5 w-3.5" /> Add worker
        </Button>
      </div>
    </div>
  );
}

// ── Deliberation mesh: a 3-column grid of equal-peer cards (same size as reconcilier
//    workers), with a dashed line connecting every pair so the visual reads as
//    "everyone talks to everyone".

function DeliberationNetworkLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd }: LayoutProps) {
  const N = steps.length;

  if (N === 0) {
    return (
      <div className="flex items-center justify-center py-12 border border-dashed rounded-lg">
        <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={onAdd}>
          <Plus className="h-3.5 w-3.5" /> Add agent
        </Button>
      </div>
    );
  }

  // Up to 3 cards per row; rows fill left-to-right, top-to-bottom.
  const cols = Math.min(N, 3);
  const numRows = Math.ceil(N / cols);

  // Card-center coordinates as percentages of the grid's bounding box.
  // Lines drawn between every pair create the peer-to-peer mesh.
  const positions = Array.from({ length: N }, (_, i) => {
    const row = Math.floor(i / cols);
    const col = i % cols;
    return {
      x: ((col + 0.5) / cols) * 100,
      y: ((row + 0.5) / numRows) * 100,
    };
  });

  return (
    <div className="space-y-3">
      <div className="relative">
        {/* Peer-to-peer mesh — dashed connector between every pair of agents */}
        {N >= 2 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none text-primary/40"
            preserveAspectRatio="none"
            aria-hidden
          >
            {positions.flatMap((p, i) =>
              positions.slice(i + 1).map((p2, j) => {
                const k = i + 1 + j;
                return (
                  <line
                    key={`${i}-${k}`}
                    x1={`${p.x}%`}
                    y1={`${p.y}%`}
                    x2={`${p2.x}%`}
                    y2={`${p2.y}%`}
                    stroke="currentColor"
                    strokeWidth="2.75"
                    strokeDasharray="6 4"
                  />
                );
              })
            )}
          </svg>
        )}

        {/* Card grid — `relative` puts cards in the same stacking layer as the
            absolutely-positioned SVG; DOM order then renders cards on top.
            `compact` cards + wider gap give the mesh room to breathe. */}
        <div
          className="grid gap-x-10 gap-y-20 relative px-4"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {steps.map((step, i) => (
            <WorkflowStepCard
              key={step.id}
              step={step}
              index={i}
              label={`Agent ${i + 1}`}
              showIndex={false}
              taskPlaceholder="What is this agent's role or viewpoint in the discussion?"
              status={statusFor(step.id, stepStatuses)}
              compact
              agents={agents}
              onUpdate={(s) => onUpdate(step.id, s)}
              onRemove={() => onRemove(step.id)}
              canRemove={steps.length > 2}
            />
          ))}
        </div>
      </div>

      <div>
        <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={onAdd}>
          <Plus className="h-3.5 w-3.5" /> Add agent
        </Button>
      </div>
    </div>
  );
}
