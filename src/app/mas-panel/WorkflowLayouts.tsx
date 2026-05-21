"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { type Agent } from "@/lib/agent-library";
import {
  type WorkflowMode,
  type WorkflowStep,
  groupLines,
  placeLineSteps,
  buildStepLabels,
} from "./workflow-types";
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
  /** Personalized mode — start a new independent line with one agent. */
  onAddLine?: () => void;
  /** Personalized mode — add an agent to a line at a specific column. */
  onAddToLine?: (line: number, slot: number) => void;
  /** Personalized mode — create edge from→to (from's output feeds to). */
  onConnect?: (fromId: string, toId: string) => void;
  /** Personalized mode — remove edge from→to. */
  onDisconnect?: (fromId: string, toId: string) => void;
  /** Page-level selected columns — drives the per-card Input DATA chips. */
  selectedCols?: string[];
}

export function WorkflowLayout(props: LayoutProps) {
  if (props.mode === "sequential") return <SequentialSLayout {...props} />;
  if (props.mode === "reconcilier") return <ReconcilierHierarchyLayout {...props} />;
  if (props.mode === "personalized") return <PersonalizedLayout {...props} />;
  return <DeliberationNetworkLayout {...props} />;
}

function statusFor(stepId: string, statuses?: Record<string, StepStatus>): StepStatus | undefined {
  return statuses?.[stepId];
}

// ── Shared connector components (dashed lines) ───────────────────────────────
// Structure mirrors ai-agents/page.tsx:172 so arrows line up with card columns.

// Width of the gap between cards. The same value is used by the horizontal
// connectors, the vertical (down) connector's gap, and the empty padding
// slots, so the snake's columns stay aligned. CONNECTOR_PX must equal the
// pixel value of the CONNECTOR_W Tailwind class (w-28 = 7rem = 112px) so the
// arrows span the gap exactly and touch the cards on both ends.
const CONNECTOR_W = "w-28";
const CONNECTOR_PX = 112;

// Scissors cursor for the "click to remove this connection" hit area. A white
// halo (drawn first, thicker) keeps it visible on any background; the black
// scissors is drawn on top. Hotspot is set at the blade crossing.
const SCISSORS_SVG =
  `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">` +
  `<g stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">` +
  `<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>` +
  `<line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/>` +
  `<line x1="8.12" y1="8.12" x2="12" y2="12"/></g>` +
  `<g stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` +
  `<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>` +
  `<line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/>` +
  `<line x1="8.12" y1="8.12" x2="12" y2="12"/></g></svg>`;
const SCISSORS_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(SCISSORS_SVG)}") 10 12, pointer`;

function SConnector({ direction, cols = 2 }: {
  direction: "left-to-right" | "right-to-left" | "down-left" | "down-right";
  /** Number of card columns in the row above, so the vertical arrow lands
   *  centered under the correct card (down-left → first, down-right → last). */
  cols?: number;
}) {
  const color = "text-muted-foreground";

  if (direction === "left-to-right") {
    return (
      <div className={`flex flex-col items-center justify-center ${CONNECTOR_W} self-center`}>
        <svg width={CONNECTOR_PX} height="28" viewBox={`0 0 ${CONNECTOR_PX} 28`} fill="none" className={color}>
          <circle cx="4" cy="14" r="4" fill="currentColor" />
          <path d={`M8 14 H${CONNECTOR_PX - 20}`} stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points={`${CONNECTOR_PX - 20},5 ${CONNECTOR_PX - 2},14 ${CONNECTOR_PX - 20},23`} fill="currentColor" />
        </svg>
      </div>
    );
  }
  if (direction === "right-to-left") {
    return (
      <div className={`flex flex-col items-center justify-center ${CONNECTOR_W} self-center`}>
        <svg width={CONNECTOR_PX} height="28" viewBox={`0 0 ${CONNECTOR_PX} 28`} fill="none" className={color}>
          <circle cx={CONNECTOR_PX - 4} cy="14" r="4" fill="currentColor" />
          <path d={`M${CONNECTOR_PX - 8} 14 H20`} stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points="20,5 2,14 20,23" fill="currentColor" />
        </svg>
      </div>
    );
  }

  // down-left / down-right — vertical arrow positioned under a card column
  const alignRight = direction === "down-right";
  const arrow = (
    <div className="flex-1 min-w-0 flex flex-col items-center">
      <svg width="28" height="56" viewBox="0 0 28 56" fill="none" className={color}>
        <circle cx="14" cy="4" r="4" fill="currentColor" />
        <path d="M14 8 V44" stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
        <polygon points="6,44 14,56 22,44" fill="currentColor" />
      </svg>
    </div>
  );
  // Mirror the row's column structure (card / gap / card / gap / …) so the
  // arrow sits centered under the first column (down-left) or last column
  // (down-right) rather than splitting the row in half.
  const spacer = <div className="flex-1 min-w-0" />;
  const arrowCol = alignRight ? cols - 1 : 0;

  return (
    <div className="flex gap-0">
      {Array.from({ length: cols }, (_, i) => (
        <React.Fragment key={i}>
          {i > 0 && <div className={`${CONNECTOR_W} shrink-0`} />}
          {i === arrowCol ? arrow : spacer}
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Sequential S-layout (3 per row, U-turn between rows) ─────────────────────

const SEQUENTIAL_COLS = 3;

function SequentialSLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd, selectedCols = [] }: LayoutProps) {
  const rows: WorkflowStep[][] = [];
  for (let i = 0; i < steps.length; i += SEQUENTIAL_COLS)
    rows.push(steps.slice(i, i + SEQUENTIAL_COLS));

  // One empty "slot" mirrors a card + its connector so partial rows keep the
  // real cards the same width as full rows above.
  const emptySlots = (n: number) =>
    Array.from({ length: n }, (_, i) => (
      <React.Fragment key={`empty-${i}`}>
        <div className="flex-1 min-w-0" />
        <div className={`${CONNECTOR_W} shrink-0`} />
      </React.Fragment>
    ));

  return (
    <div className="space-y-0">
      {rows.map((row, rowIdx) => {
        const isEvenRow = rowIdx % 2 === 0;
        const orderedRow = isEvenRow ? row : [...row].reverse();
        const globalBase = rowIdx * SEQUENTIAL_COLS;
        const missing = SEQUENTIAL_COLS - row.length;

        return (
          <div key={rowIdx}>
            {/* Row of cards */}
            <div className="flex gap-0 items-stretch">
              {/* Partial odd row: empty slots on the LEFT so cards sit on the right */}
              {missing > 0 && !isEvenRow && emptySlots(missing)}
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
                        showInputData
                        inputCols={selectedCols}
                        prevStepLabel={globalIdx > 0 ? `Step ${globalIdx}` : null}
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
              {/* Partial even row: empty slots on the RIGHT so cards sit on the left */}
              {missing > 0 && isEvenRow && emptySlots(missing)}
            </div>

            {/* U-turn connector between rows — lands under the card where the S
                turns. Only the last row can be partial, and partial rows keep
                their cards aligned to the snake's entry edge, so the turn
                direction depends only on the current row's parity. */}
            {rowIdx < rows.length - 1 && (
              <SConnector
                direction={isEvenRow ? "down-right" : "down-left"}
                cols={SEQUENTIAL_COLS}
              />
            )}
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

function ReconcilierHierarchyLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd, selectedCols = [] }: LayoutProps) {
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
              label="Manager (top of tree)"
              showIndex={false}
              taskPlaceholder="How should the manager combine the workers' outputs into a final answer?"
              samplePrompts={SAMPLE_RECONCILER_PROMPTS}
              status={statusFor(reconciler.id, stepStatuses)}
              agents={agents}
              showInputData
              inputCols={selectedCols}
              staticSources={["Workers' outputs"]}
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
              // Workers render in a wrapping grid of `gridCols` columns (max 3),
              // so worker i sits in column `i % gridCols`. Aim each arrow at its
              // column. The center column points straight up to the manager and
              // stacked workers there share that one arrow; in the left/right
              // columns, stacked workers are fanned apart by a small offset so
              // each gets its own distinct (still clearly side-leaning) arrow.
              const gridCols = Math.min(workers.length, 3);
              const col = i % gridCols;
              const colCenter = ((col + 0.5) / gridCols) * 100;
              const isCenterCol = gridCols % 2 === 1 && col === (gridCols - 1) / 2;
              const countInCol = Math.ceil((workers.length - col) / gridCols);
              const rowInCol = Math.floor(i / gridCols);
              let xPct = colCenter;
              if (!isCenterCol && countInCol > 1) {
                const spacing = 8;
                const offset = (rowInCol - (countInCol - 1) / 2) * spacing;
                xPct = Math.min(96, Math.max(4, colCenter + offset));
              }
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
              showInputData
              inputCols={selectedCols}
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

// ── Personalized: free-form DAG of agents grouped into visual lines ──────────

interface Rect { x: number; y: number; w: number; h: number; }

function PersonalizedLayout({
  steps,
  agents,
  stepStatuses,
  onUpdate,
  onRemove,
  onAddLine,
  onAddToLine,
  onConnect,
  onDisconnect,
  selectedCols = [],
}: LayoutProps) {
  // Lines are visual rows only — data flow is the explicit `inputs` edges.
  const lines = groupLines(steps);
  const stepLabels = buildStepLabels(steps);

  const edges: { from: string; to: string }[] = [];
  for (const s of steps) for (const src of s.inputs ?? []) edges.push({ from: src, to: s.id });

  const [connectingFrom, setConnectingFrom] = React.useState<string | null>(null);
  const canvasRef = React.useRef<HTMLDivElement | null>(null);
  const cardRefs = React.useRef<Map<string, HTMLDivElement>>(new Map());
  const [rects, setRects] = React.useState<Record<string, Rect>>({});
  const [resizeTick, setResizeTick] = React.useState(0);

  // Esc cancels an in-progress connection.
  React.useEffect(() => {
    if (!connectingFrom) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setConnectingFrom(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [connectingFrom]);

  // Drop a dangling connect if its source step was removed mid-gesture.
  React.useEffect(() => {
    if (connectingFrom && !steps.some((s) => s.id === connectingFrom)) {
      setConnectingFrom(null);
    }
  }, [steps, connectingFrom]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => setResizeTick((t) => t + 1));
    ro.observe(canvas);
    const onWin = () => setResizeTick((t) => t + 1);
    window.addEventListener("resize", onWin);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", onWin);
    };
  }, []);
  // Re-measure when the graph shape changes (signature) or the container is
  // resized (resizeTick — also fires when a card grows from typing, via the
  // ResizeObserver). The equality guard avoids redundant state churn.
  const signature =
    steps
      .map((s) => `${s.id}:${s.line ?? 0}:${(s.inputs ?? []).join("|")}`)
      .join(",") + `#${steps.length}`;
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cRect = canvas.getBoundingClientRect();
    const next: Record<string, Rect> = {};
    cardRefs.current.forEach((el, id) => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      next[id] = { x: r.left - cRect.left, y: r.top - cRect.top, w: r.width, h: r.height };
    });
    setRects((prev) => {
      const same =
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(next).every(
          (k) =>
            prev[k] &&
            prev[k].x === next[k].x &&
            prev[k].y === next[k].y &&
            prev[k].w === next[k].w &&
            prev[k].h === next[k].h,
        );
      return same ? prev : next;
    });
  }, [signature, resizeTick]);

  const setCardRef = (id: string) => (el: HTMLDivElement | null) => {
    if (el) cardRefs.current.set(id, el);
    else cardRefs.current.delete(id);
  };

  const completeConnect = (targetId: string) => {
    if (!connectingFrom) return;
    if (connectingFrom !== targetId) onConnect?.(connectingFrom, targetId);
    setConnectingFrom(null);
  };

  function edgePath(from: string, to: string): string | null {
    const a = rects[from];
    const b = rects[to];
    if (!a || !b) return null;
    const sx = a.x + a.w;
    const sy = a.y + a.h / 2;
    const tx = b.x;
    const ty = b.y + b.h / 2;
    const dx = Math.max(40, Math.abs(tx - sx) * 0.5);
    return `M ${sx} ${sy} C ${sx + dx} ${sy}, ${tx - dx} ${ty}, ${tx} ${ty}`;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div className="text-xs text-muted-foreground">
          {connectingFrom
            ? "Connecting… click a target agent (Esc to cancel)"
            : "Click an agent’s ● output dot, then click another agent to connect. Click an arrow to remove it."}
        </div>
        <Button
          variant="outline"
          size="sm"
          className="text-xs gap-1.5 shrink-0"
          onClick={() => onAddLine?.()}
        >
          <Plus className="h-3.5 w-3.5" /> Add AI Agent line
        </Button>
      </div>

      {lines.length === 0 ? (
        <div className="flex items-center justify-center py-12 border border-dashed rounded-lg text-sm text-muted-foreground">
          No agents yet — click “+ Add AI Agent line” to start a line.
        </div>
      ) : (
        <div ref={canvasRef} className="relative space-y-6">
          {/* Curved connection overlay. Sits ABOVE the cards (z-30) so the
              whole arrow — and its scissors hit area — is exposed along its
              full length, not just the gap between cards. The <svg> keeps
              pointer-events-none; only the thin curve ribbon below is
              clickable, so card content stays interactive everywhere else. */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-30"
            style={{ overflow: "visible" }}
            aria-hidden
          >
            <defs>
              <marker
                id="pz-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0 0 L10 5 L0 10 z" fill="currentColor" />
              </marker>
            </defs>
            {edges.map((e, i) => {
              const d = edgePath(e.from, e.to);
              if (!d) return null;
              return (
                <g key={`${e.from}->${e.to}-${i}`} className="text-primary">
                  <path
                    d={d}
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="6 4"
                    fill="none"
                    markerEnd="url(#pz-arrow)"
                  />
                  <path
                    d={d}
                    stroke="transparent"
                    strokeWidth="22"
                    fill="none"
                    strokeLinecap="round"
                    className="pointer-events-auto"
                    style={{ cursor: SCISSORS_CURSOR }}
                    onClick={() => onDisconnect?.(e.from, e.to)}
                  >
                    <title>Click to remove this connection</title>
                  </path>
                </g>
              );
            })}
          </svg>

          {lines.map(([lineNo, lineSteps], li) => {
            const placed = placeLineSteps(lineSteps);
            return (
              <div key={lineNo} className="relative">
                <div className="flex items-stretch">
                  {placed.map((step, slot) => (
                    <div
                      key={step?.id ?? `slot-${slot}`}
                      className="flex-1 min-w-0 flex px-10"
                    >
                      {step ? (
                        <div ref={setCardRef(step.id)} className="relative w-full">
                          <WorkflowStepCard
                            step={step}
                            index={slot}
                            label={`Line ${li + 1} · Agent ${slot + 1}`}
                            showIndex={false}
                            compact
                            status={statusFor(step.id, stepStatuses)}
                            agents={agents}
                            showInputData
                            inputCols={selectedCols}
                            connectedSources={(step.inputs ?? []).map((srcId) => ({
                              id: srcId,
                              label: stepLabels[srcId] ?? srcId,
                            }))}
                            onUpdate={(s) => onUpdate(step.id, s)}
                            onRemove={() => onRemove(step.id)}
                            canRemove
                          />

                          {/* Input handle (left). Hidden until the user clicks
                              another card's ● output dot — then it appears on
                              every OTHER card as a connection target. */}
                          {connectingFrom && connectingFrom !== step.id && (
                            <button
                              type="button"
                              title="Connect into this agent"
                              onClick={(ev) => {
                                ev.stopPropagation();
                                completeConnect(step.id);
                              }}
                              className="absolute -left-2.5 top-1/2 -translate-y-1/2 z-40 h-5 w-5 rounded-[4px] border-2 bg-background transition-colors border-primary ring-2 ring-primary/30 animate-pulse cursor-pointer"
                            />
                          )}

                          {/* Output handle (right) — click to start a
                              connection from this card. */}
                          <button
                            type="button"
                            title="Start a connection from this agent"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              setConnectingFrom((cur) =>
                                cur === step.id ? null : step.id,
                              );
                            }}
                            className={`absolute -right-2 top-1/2 -translate-y-1/2 z-40 h-5 w-5 rounded-full border-2 bg-background flex items-center justify-center transition-colors ${
                              connectingFrom === step.id
                                ? "border-primary ring-2 ring-primary/30"
                                : "border-muted-foreground/50 hover:border-primary"
                            }`}
                          >
                            <span className="h-2 w-2 rounded-full bg-primary" />
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onAddToLine?.(lineNo, slot)}
                          title="Add an agent here"
                          className="flex-1 min-h-[7rem] rounded-md border border-dashed border-muted-foreground/40 flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-muted-foreground/70 hover:bg-muted/40 transition-colors"
                        >
                          <Plus className="h-5 w-5" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
