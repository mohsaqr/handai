"use client";

import React from "react";
import { Link2, Plus } from "lucide-react";
import { type Agent } from "@/lib/agent-library";
import {
  type WorkflowMode,
  type WorkflowStep,
} from "./workflow-types";
import { WorkflowStepCard, type StepStatus } from "./WorkflowStepCard";

interface LayoutProps {
  mode: WorkflowMode;
  steps: WorkflowStep[];
  agents: Agent[];
  stepStatuses?: Record<string, StepStatus>;
  onUpdate: (id: string, step: WorkflowStep) => void;
  onRemove: (id: string) => void;
  /** Personalized mode — start a new independent line with one agent. */
  onAddLine?: () => void;
  /** Personalized mode — add an agent to a line at a specific column. */
  onAddToLine?: (line: number, slot: number) => void;
  /** Personalized mode — create edge from→to (from's output feeds to). */
  onConnect?: (fromId: string, toId: string) => void;
  /** Personalized mode — remove edge from→to. */
  onDisconnect?: (fromId: string, toId: string) => void;
  /** Reconcilier mode — cut a worker→Judge spoke (scissors). */
  onCutJudge?: (workerId: string) => void;
  /** Reconcilier mode — restore a cut worker→Judge spoke (click worker dot → Judge). */
  onRestoreJudge?: (workerId: string) => void;
  /** Change a step's agent; permutes it with the picked agent's step so nothing is
   *  duplicated or lost. Used by Sequential steps and the Judge card (which passes
   *  its own step id). */
  onSwapStepAgent?: (stepId: string, agentId: string) => void;
  /** Page-level selected columns — the per-card Input DATA chips shown by default. */
  selectedCols?: string[];
  /** Every column in the uploaded file. Columns here that aren't in
   *  `selectedCols` are offered in each card's "+ column" picker, so a card can
   *  pull in any uploaded column, not only the globally-defined ones. */
  allCols?: string[];
  /** Name of an uploaded unstructured file (PDF/DOCX/TXT). Lets cards show a
   *  document chip instead of "no input" when they receive the file's text. */
  documentInput?: string;
  /** Personalized mode — number of agent lines to show (lines may be empty).
   *  "Add AI Agent line" increments this. */
  lineCount?: number;
}

export function WorkflowLayout(props: LayoutProps) {
  if (props.mode === "sequential") return <SequentialSLayout {...props} />;
  // Judge, Individual and Deliberation all use one radial ring of agents. They
  // differ only in: a center hub (Judge), editable directional edges (Individual),
  // or a read-only all-to-all mesh with no Connect (Deliberation).
  return <RadialWorkflowLayout {...props} />;
}

function statusFor(stepId: string, statuses?: Record<string, StepStatus>): StepStatus | undefined {
  return statuses?.[stepId];
}

// ── Shared connector components (dashed lines) ───────────────────────────────
// Structure mirrors ai-agents/page.tsx:172 so arrows line up with card columns.

// Maximum gap between cards, in px. The Sequential layout measures its container
// and reserves this same width (or less, on a narrow panel) for the horizontal
// connectors, the vertical (down) connector's gap, and the empty padding slots,
// so the snake's columns stay aligned and the arrows span the gap exactly. A wide
// gap keeps the step cards compact and gives the connecting arrows room to read;
// on a narrow screen the gap shrinks so the cards don't get crushed.
const CONNECTOR_PX = 384;

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

function SConnector({ direction, cols = 2, widthPx = CONNECTOR_PX }: {
  direction: "left-to-right" | "right-to-left" | "down-left" | "down-right";
  /** Number of card columns in the row above, so the vertical arrow lands
   *  centered under the correct card (down-left → first, down-right → last). */
  cols?: number;
  /** Measured gap width in px — the same value the row reserves between cards, so
   *  the arrow spans the gap exactly. Shrinks on narrow screens (responsive). */
  widthPx?: number;
}) {
  const color = "text-muted-foreground";

  if (direction === "left-to-right") {
    return (
      <div className="flex flex-col items-center justify-center self-center shrink-0" style={{ width: widthPx }}>
        <svg width={widthPx} height="28" viewBox={`0 0 ${widthPx} 28`} fill="none" className={color}>
          <circle cx="4" cy="14" r="4" fill="currentColor" />
          <path d={`M8 14 H${widthPx - 20}`} stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points={`${widthPx - 20},5 ${widthPx - 2},14 ${widthPx - 20},23`} fill="currentColor" />
        </svg>
      </div>
    );
  }
  if (direction === "right-to-left") {
    return (
      <div className="flex flex-col items-center justify-center self-center shrink-0" style={{ width: widthPx }}>
        <svg width={widthPx} height="28" viewBox={`0 0 ${widthPx} 28`} fill="none" className={color}>
          <circle cx={widthPx - 4} cy="14" r="4" fill="currentColor" />
          <path d={`M${widthPx - 8} 14 H20`} stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
          <polygon points="20,5 2,14 20,23" fill="currentColor" />
        </svg>
      </div>
    );
  }

  // down-left / down-right — vertical arrow positioned under a card column
  const alignRight = direction === "down-right";
  const arrow = (
    <div className="flex-1 min-w-0 flex flex-col items-center">
      <svg width="28" height="72" viewBox="0 0 28 72" fill="none" className={color}>
        <circle cx="14" cy="4" r="4" fill="currentColor" />
        <path d="M14 8 V60" stroke="currentColor" strokeWidth="3.5" strokeDasharray="6 4" />
        <polygon points="6,60 14,72 22,60" fill="currentColor" />
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
          {i > 0 && <div className="shrink-0" style={{ width: widthPx }} />}
          {i === arrowCol ? arrow : spacer}
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Sequential S-layout (3 per row, U-turn between rows) ─────────────────────

const SEQUENTIAL_COLS = 3;

function SequentialSLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onSwapStepAgent, selectedCols = [], allCols = [], documentInput }: LayoutProps) {
  // Sequential cards are a pure projection of the Configure Agents pool — one card
  // per agent, in pool order — so there's no on-canvas add/remove (manage agents
  // in Configure Agents). Picking a different agent in a card's dropdown SWAPS the
  // two agents' positions (onSwapStepAgent), letting the user re-order the pipeline
  // without changing the agent set.

  // Measure the live container width so the snake stays readable at any size: the
  // number of cards per row AND the connector (arrow) width both shrink as the
  // panel narrows, instead of fixed columns crushing the cards. A card needs about
  // CARD_MIN px to stay usable, so we fit as many whole cards (+ gaps) as the width
  // allows, then spend any leftover space widening the arrows (capped at 384px).
  const wrapRef = React.useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = React.useState(0);
  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(e.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const CARD_MIN = 276; // comfortable card width (avatar + dropdown + chips)
  // ≥3 cards need 3·CARD_MIN + 2 gaps; ≥2 need 2·CARD_MIN + 1 gap; else stack to 1.
  const cols =
    width === 0 ? Math.min(SEQUENTIAL_COLS, Math.max(1, steps.length))
    : width >= CARD_MIN * 3 + 80 ? SEQUENTIAL_COLS
    : width >= CARD_MIN * 2 + 40 ? 2
    : 1;
  // Connector spans the leftover width after every card gets CARD_MIN, capped at
  // 384 (the wide-screen "long arrow" look). 0 when single-column (no side arrows).
  // Before the first measurement (width === 0, i.e. SSR / first paint) use a modest
  // gap so a narrow container can't overflow for the one frame before the
  // ResizeObserver fires and swaps in the real value.
  const connectorPx =
    cols <= 1
      ? 0
      : width === 0
        ? 144
        : Math.round(Math.max(40, Math.min(CONNECTOR_PX, (width - cols * CARD_MIN) / (cols - 1))));

  const cellCount = steps.length;
  const rows: number[][] = [];
  for (let i = 0; i < cellCount; i += cols) {
    const row: number[] = [];
    for (let j = i; j < Math.min(i + cols, cellCount); j++) row.push(j);
    rows.push(row);
  }

  // One empty "slot" mirrors a card + its connector so partial rows keep the
  // real cards the same width as full rows above.
  const emptySlots = (n: number) =>
    Array.from({ length: n }, (_, i) => (
      <React.Fragment key={`empty-${i}`}>
        <div className="flex-1 min-w-0" />
        <div className="shrink-0" style={{ width: connectorPx }} />
      </React.Fragment>
    ));

  // Mirror the radial modes' empty state: with no agents there are no cards and no
  // on-canvas add, so point the user to Configure Agents.
  if (steps.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 border border-dashed rounded-lg text-sm text-muted-foreground text-center px-6">
        Add agents in “Configure Agents” above — they appear here automatically.
      </div>
    );
  }

  return (
    <div ref={wrapRef} className="space-y-0">
      {rows.map((row, rowIdx) => {
        const isEvenRow = rowIdx % 2 === 0;
        const orderedRow = isEvenRow ? row : [...row].reverse();
        const missing = cols - row.length;

        return (
          <div key={rowIdx}>
            {/* Row of cards */}
            <div className="flex gap-0 items-stretch">
              {/* Partial odd row: empty slots on the LEFT so cards sit on the right */}
              {missing > 0 && !isEvenRow && emptySlots(missing)}
              {orderedRow.map((globalIdx, colIdx) => {
                const isLastInRow = colIdx === orderedRow.length - 1;
                return (
                  <React.Fragment key={steps[globalIdx].id}>
                    <div className="flex-1 min-w-0">
                      <WorkflowStepCard
                        step={steps[globalIdx]}
                        index={globalIdx}
                        label={`Step ${globalIdx + 1}`}
                        status={statusFor(steps[globalIdx].id, stepStatuses)}
                        agents={agents}
                        compact
                        hideAgentLabel
                        showInputData
                        inputCols={selectedCols}
                        allCols={allCols}
                        documentInput={documentInput}
                        prevStepLabel={globalIdx > 0 ? `Step ${globalIdx}` : null}
                        lockAgent
                        onSwapAgent={(agentId) => onSwapStepAgent?.(steps[globalIdx].id, agentId)}
                        onUpdate={(s) => onUpdate(steps[globalIdx].id, s)}
                        onRemove={() => onRemove(steps[globalIdx].id)}
                        canRemove={false}
                      />
                    </div>
                    {!isLastInRow && (
                      <SConnector
                        direction={isEvenRow ? "left-to-right" : "right-to-left"}
                        widthPx={connectorPx}
                      />
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
                cols={cols}
                widthPx={connectorPx}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Radial layout: agents arranged in a ring. Judge mode adds a center hub the
//    workers feed; Individual mode uses the same ring with no hub. ─────────────

interface Rect { x: number; y: number; w: number; h: number; }

function RadialWorkflowLayout({ mode, steps, agents, stepStatuses, onUpdate, onRemove, onConnect, onDisconnect, onCutJudge, onRestoreJudge, onSwapStepAgent, selectedCols = [], allCols = [], documentInput }: LayoutProps) {
  // Three modes share this radial ring:
  //  • Judge (reconcilier): a center hub (steps[0]) every worker feeds.
  //  • Individual (personalized): no hub; nodes wired by explicit agent→agent edges.
  //  • Deliberation: no hub; NOT user-wired — every agent implicitly talks to every
  //    other, drawn as a full mesh of plain lines, with no "Connect" UI.
  // `hasHub` gates the Judge-only pieces; `meshMode` swaps the editable directional
  // edges for the read-only all-to-all mesh and hides the connection controls.
  const hasHub = mode === "reconcilier";
  const meshMode = mode === "deliberation";
  const allowConnect = !meshMode;
  const reconciler = hasHub ? steps[0] : null;
  const workers = hasHub ? steps.slice(1) : steps;
  const workerIds = new Set(workers.map((w) => w.id));
  const nodeNoun = hasHub ? "worker" : "agent";
  const nodeName = (i: number) =>
    hasHub ? `Worker ${i + 1}` : meshMode ? `Participant ${i + 1}` : `Agent ${i + 1}`;
  // Label for a node's "<source>'s output" Input DATA chip: prefer the assigned
  // agent's real name (e.g. "Manager 1"), falling back to the positional label
  // ("Worker N" / "Agent N") when the agent is unnamed or unassigned.
  const displayName = (step: WorkflowStep, i: number) => {
    const a = step.agentId ? agents.find((x) => x.id === step.agentId) : null;
    return a?.name?.trim() || nodeName(i);
  };

  // Worker→Judge spokes the user has cut live on the Judge step as
  // `judgeExcluded`; a worker not listed there feeds the Judge (the default).
  const judgeExcluded = new Set(reconciler?.judgeExcluded ?? []);
  const connectedWorkers = workers.filter((w) => !judgeExcluded.has(w.id));
  const allConnected = connectedWorkers.length === workers.length;
  // The Judge's Input DATA chip(s): the collective "Workers' outputs" when every
  // worker still feeds it, otherwise one chip per still-connected worker.
  const judgeSources =
    workers.length === 0
      ? []
      : allConnected
        ? ["Workers' outputs"]
        : connectedWorkers.map((w) => `${displayName(w, workers.indexOf(w))}'s output`);

  // Measure the real card positions so each spoke can run from its worker's
  // bottom edge straight to the Judge's top edge. A fixed-height band with
  // percentage x-positions breaks once workers wrap onto a second row — the
  // lines start floating below the cards instead of touching them. Measuring
  // the actual DOM rects keeps the arrows anchored to
  // the cards no matter how the grid wraps.
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const workerRefs = React.useRef<Map<string, HTMLDivElement>>(new Map());
  const judgeRef = React.useRef<HTMLDivElement | null>(null);
  const [rects, setRects] = React.useState<{
    workers: Record<string, Rect>;
    judge: Rect | null;
  }>({ workers: {}, judge: null });
  const [resizeTick, setResizeTick] = React.useState(0);
  // The circular canvas is a square sized to the container's real width. We set
  // the height explicitly (rather than relying on `aspect-ratio`) so the box can
  // never collapse to zero height — which would pile every absolutely-positioned
  // card on top of each other instead of spreading them around the circle.
  const [boxSize, setBoxSize] = React.useState(560);

  // Re-measure whenever the graph shape OR any card's content changes. The
  // content matters because a card's height grows when it gains an input chip
  // (on connect) or a longer agent name — and the spokes/arrows must follow the
  // card's new edges, not its old ones.
  const signature =
    steps
      .map((s) => `${s.id}:${s.agentId ?? ""}:${(s.inputs ?? []).join("|")}`)
      .join(",") + `#${steps.length}`;

  React.useEffect(() => {
    const ro = new ResizeObserver(() => setResizeTick((t) => t + 1));
    if (containerRef.current) ro.observe(containerRef.current);
    // Observe each card too. Cards are absolutely positioned, so when one grows
    // (e.g. gains an input chip on connect) the container's size doesn't change
    // and a container-only observer would never fire — leaving the arrows pinned
    // to the card's stale rect. Watching the cards themselves keeps every spoke
    // and worker→worker arrow locked onto the live card edges.
    workerRefs.current.forEach((el) => ro.observe(el));
    if (judgeRef.current) ro.observe(judgeRef.current);
    const onWin = () => setResizeTick((t) => t + 1);
    window.addEventListener("resize", onWin);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", onWin);
    };
  }, [signature]);

  React.useLayoutEffect(() => {
    const canvas = containerRef.current;
    if (!canvas) return;
    const cRect = canvas.getBoundingClientRect();
    // Keep the canvas square: height tracks the real (parent-driven) width.
    if (cRect.width > 0) {
      setBoxSize((prev) => (Math.abs(prev - cRect.width) < 0.5 ? prev : cRect.width));
    }
    const toRect = (el: HTMLElement): Rect => {
      const r = el.getBoundingClientRect();
      return { x: r.left - cRect.left, y: r.top - cRect.top, w: r.width, h: r.height };
    };
    const nextWorkers: Record<string, Rect> = {};
    workerRefs.current.forEach((el, id) => {
      if (el) nextWorkers[id] = toRect(el);
    });
    const nextJudge = judgeRef.current ? toRect(judgeRef.current) : null;
    setRects((prev) => {
      const eq = (a: Rect, b: Rect) =>
        a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
      const sameWorkers =
        Object.keys(nextWorkers).length === Object.keys(prev.workers).length &&
        Object.keys(nextWorkers).every(
          (k) => prev.workers[k] && eq(prev.workers[k], nextWorkers[k]),
        );
      const sameJudge =
        (!nextJudge && !prev.judge) ||
        (!!nextJudge && !!prev.judge && eq(nextJudge, prev.judge));
      return sameWorkers && sameJudge ? prev : { workers: nextWorkers, judge: nextJudge };
    });
    // `boxSize` MUST be a dependency: this effect both reads it (cards are placed
    // from it via workerPos) and sets it. On a window resize, resizeTick fires the
    // effect once — it measures the cards at their OLD positions and schedules the
    // new boxSize. That new boxSize repositions every card, so the effect has to run
    // AGAIN to re-measure them at their new spots; without boxSize here it wouldn't,
    // and the spokes/arrows would stay frozen at the pre-resize (small) geometry
    // until something else (e.g. reconnecting a card) changed `signature`. The
    // setBoxSize guard (|Δ| < 0.5 → no-op) and the rect-equality guard stop this
    // from looping.
  }, [signature, resizeTick, boxSize]);

  const setWorkerRef = (id: string) => (el: HTMLDivElement | null) => {
    if (el) workerRefs.current.set(id, el);
    else workerRefs.current.delete(id);
  };

  // ── Worker ↔ worker connections (same interactive system as personalized) ──
  // The Judge → workers link is PERMANENT (every worker always feeds the Judge),
  // drawn automatically below. Workers can additionally be wired to each other:
  // an edge from worker A to worker B means B receives A's output and runs after
  // it. Those edges live on the target worker's `inputs` and use the same
  // click-dot / arrow / scissors UI as the personalized layout.
  const workerEdges: { from: string; to: string }[] = [];
  for (const w of workers)
    for (const src of w.inputs ?? [])
      if (workerIds.has(src)) workerEdges.push({ from: src, to: w.id });

  const [connectingFrom, setConnectingFrom] = React.useState<string | null>(null);

  // Esc cancels an in-progress connection.
  React.useEffect(() => {
    if (!connectingFrom) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setConnectingFrom(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [connectingFrom]);

  // Drop a dangling connect if its source worker was removed mid-gesture.
  React.useEffect(() => {
    if (connectingFrom && !steps.some((s) => s.id === connectingFrom)) {
      setConnectingFrom(null);
    }
  }, [connectingFrom, steps]);

  const completeConnect = (targetId: string) => {
    if (!connectingFrom) return;
    if (connectingFrom !== targetId) onConnect?.(connectingFrom, targetId);
    setConnectingFrom(null);
  };

  // ── Radial layout geometry ──────────────────────────────────────────────
  // Workers sit at the vertices of a regular polygon around the Judge hub —
  // triangle for 3, square for 4, pentagon for 5, and so on. Rather than a fixed
  // radius, each worker is pushed as far toward the canvas edge as its card
  // allows (independently per axis), so the ring fills the whole width/height and
  // the Judge→worker spokes are as long as possible.
  const N = workers.length;
  // Canvas width = the measured container width (height is capped below).
  const canvasW = boxSize;
  // Card width ADAPTS to the ring so cards never pile on top of one another on a
  // narrow panel — it's the largest width that keeps the layout collision-free for
  // the current canvas width and node count, capped at the full 400/410px on a
  // roomy panel and floored so cards stay readable. Two limits, whichever bites:
  //   • worker ↔ Judge hub (Judge mode only): the ring radius is bounded by the
  //     canvas, which stops a worker reaching the centre hub once its width passes
  //     ~canvasW/3.
  //   • neighbour ↔ neighbour: adjacent vertices of the N-gon are 2·R·sin(π/N)
  //     apart, and that gap must stay wider than a card.
  // ×0.92 leaves a small breathing gap instead of letting cards merely touch. One
  // node (or none) has no ring to crowd, so it keeps the full width.
  const ringSafeW = (() => {
    if (N <= 1) return 400;
    const hubLimit = hasHub ? canvasW / 3 : Infinity;
    const s = Math.sin(Math.PI / N);
    const neighbourLimit = (canvasW * s) / (1 + s);
    return Math.min(hubLimit, neighbourLimit) * 0.92;
  })();
  // Floors match the card's intrinsic minimum — the w-36 (144px) avatar plus the
  // card's padding + gap — so a card never shrinks past its avatar (which would
  // overflow) and the collision math above stays accurate. Below the canvas width
  // where even a floored ring fits, a multi-card ring simply can't avoid overlap.
  const WORKER_W = Math.round(Math.min(400, Math.max(175, ringSafeW)));
  const JUDGE_W = Math.round(Math.min(410, Math.max(180, ringSafeW)));
  // Spread the ring wider as workers multiply. With 7+ workers a 1700-wide canvas
  // crowds the cards into the middle (so the Judge→worker spokes shrink to almost
  // nothing) while empty space sits on the left/right of the page. Raising the
  // canvas cap lets the cards push out into that space — the cards already hug the
  // canvas margins, so a wider canvas == cards further apart == longer arrows.
  // Few workers stay compact so they don't drift absurdly far apart.
  const CANVAS_MAX_W = N >= 9 ? 2800 : N >= 7 ? 2400 : N >= 5 ? 2000 : 1700;
  const CANVAS_MAX_H = N >= 7 ? 860 : 720;
  // Orientation: pointy-top by default (triangle/pentagon point up). Two workers
  // flank the Judge left/right; four sit at the corners of an upright square.
  const startDeg = N === 2 ? 180 : N === 4 ? -45 : -90;
  const angles = Array.from(
    { length: N },
    (_, i) => ((startDeg + (360 / Math.max(N, 1)) * i) * Math.PI) / 180,
  );

  // A lone worker stacks vertically above the Judge and needs real vertical room
  // for a visible spoke. Two workers fan across the TOP with the Judge centered
  // below (an inverted "∨" that funnels both workers down into the Judge) rather
  // than a flat worker–Judge–worker line, so they need the same taller two-row box.
  // Only the Judge mode's lone worker / pair stack above the hub; with no hub the
  // nodes just sit around the ring.
  const isVerticalStack = hasHub && N === 1;
  const isTopFan = hasHub && N === 2;
  const canvasH = isVerticalStack
    ? Math.min(boxSize, 500)
    : isTopFan
      ? Math.min(boxSize, 480)
      : Math.min(boxSize, N >= 3 ? CANVAS_MAX_H : 320);
  // The worker with the largest |cos| (resp. |sin|) is the one closest to the
  // left/right (resp. top/bottom) edge. Size the horizontal/vertical radius so
  // THAT worker just clears the margin — every other worker then sits inside.
  const MARGIN = 10;
  const CARD_HALF_H = 95; // ≈ half a tall worker card (avatar + a few input chips)
  const maxCos = angles.reduce((m, a) => Math.max(m, Math.abs(Math.cos(a))), 0.001);
  const maxSin = angles.reduce((m, a) => Math.max(m, Math.abs(Math.sin(a))), 0.001);
  const rx = Math.max(0, canvasW / 2 - WORKER_W / 2 - MARGIN) / maxCos;
  const ry = Math.max(0, canvasH / 2 - CARD_HALF_H - MARGIN) / maxSin;
  // Top-fan half-spread: how far each of the two workers sits left/right of
  // centre. Bounded so the pair never overflows the canvas, and capped at
  // ¾·card-width so a roomy canvas keeps the triangle compact instead of
  // flattening the funnel back toward a horizontal line.
  const fanHalfX = Math.min(
    Math.max(0, canvasW / 2 - WORKER_W / 2 - MARGIN),
    WORKER_W * 0.75,
  );
  const workerPos = (i: number) => {
    // Single worker: pin it near the top, centered over the Judge, so the spoke
    // runs straight down. The radial formula would place it only ~ry above the
    // centered Judge — too little to clear the Judge's own height, so the cards
    // would overlap with no room for the arrow.
    if (isVerticalStack) {
      return { leftPx: canvasW / 2, topPx: MARGIN + CARD_HALF_H };
    }
    // Two workers: side by side along the top band (i=0 left, i=1 right), both
    // funneling down into the Judge centered below — an inverted triangle.
    if (isTopFan) {
      return {
        leftPx: canvasW / 2 + (i === 0 ? -fanHalfX : fanHalfX),
        topPx: MARGIN + CARD_HALF_H,
      };
    }
    return {
      leftPx: canvasW / 2 + rx * Math.cos(angles[i]),
      topPx: canvasH / 2 + ry * Math.sin(angles[i]),
    };
  };
  // Judge sits at the canvas center for the radial layouts; for the lone-worker
  // vertical stack and the two-worker top fan it drops to the bottom so the
  // worker→Judge spokes span the canvas instead of being squeezed into one half.
  const judgeTopPx =
    isVerticalStack || isTopFan ? canvasH - MARGIN - CARD_HALF_H : canvasH / 2;

  const rectCenter = (r: Rect) => ({ x: r.x + r.w / 2, y: r.y + r.h / 2 });
  // Point on a rect's boundary in the direction of (tx, ty) — anchors each
  // spoke/edge to the card's edge instead of its hidden center.
  function rectEdgeToward(r: Rect, tx: number, ty: number): { x: number; y: number } {
    const cx = r.x + r.w / 2;
    const cy = r.y + r.h / 2;
    const dx = tx - cx;
    const dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    const sx = dx === 0 ? Infinity : r.w / 2 / Math.abs(dx);
    const sy = dy === 0 ? Infinity : r.h / 2 / Math.abs(dy);
    const s = Math.min(sx, sy);
    return { x: cx + dx * s, y: cy + dy * s };
  }

  // Worker → worker edge: a quadratic from the source card's edge to the target's,
  // bowed perpendicular to the chord and away from the Judge hub so it arcs around
  // the card instead of across it.
  function workerEdgePath(from: string, to: string): string | null {
    const a = rects.workers[from];
    const b = rects.workers[to];
    if (!a || !b) return null;
    const ac = rectCenter(a);
    const bc = rectCenter(b);
    const start = rectEdgeToward(a, bc.x, bc.y);
    const end = rectEdgeToward(b, ac.x, ac.y);
    // Route the edge AROUND the Judge: bow it perpendicular to the chord, to the
    // side AWAY from the Judge hub. That single rule does everything — a connection
    // always curves around whichever side of the Judge it already leans toward (so a
    // bottom-left→top worker swings up the left, never across the card), and it can
    // never bow toward the card because the bow points away from where the card is.
    // With no hub (Individual) it bows away from the canvas centre for the same look.
    const hub = rects.judge ? rectCenter(rects.judge) : { x: canvasW / 2, y: canvasH / 2 };
    const mx = (start.x + end.x) / 2;
    const my = (start.y + end.y) / 2;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const len = Math.hypot(dx, dy) || 1;
    // Unit normal to the chord, flipped so it points away from the hub.
    let nx = -dy / len;
    let ny = dx / len;
    const offset = (hub.x - mx) * nx + (hub.y - my) * ny; // hub's signed offset along it
    if (offset > 0) {
      nx = -nx;
      ny = -ny;
    }
    const dist = Math.abs(offset); // perpendicular distance hub → chord
    // Degenerate case (workers exactly opposite, hub sitting on the chord): "away"
    // is undefined, so bow downward rather than risk riding over the top.
    if (dist < 2 && ny < 0) {
      nx = -nx;
      ny = -ny;
    }
    // Gentle by default; widen only as much as the Judge box pokes past the chord on
    // this side, so distant edges curve lightly and close ones swing wide enough to
    // clear. A quadratic's apex reaches half its control offset, hence the ×2. The
    // box reach is measured live, so the arc adapts as the Judge card resizes.
    let bow = 36;
    if (rects.judge) {
      const reach = Math.abs(nx) * (rects.judge.w / 2) + Math.abs(ny) * (rects.judge.h / 2);
      bow = Math.max(36, 2 * (reach - dist + 28));
    }
    const cx = mx + nx * bow;
    const cy = my + ny * bow;
    // End exactly on the target card's edge so the arrow reaches the card. The
    // endpoint is recomputed from the live rect each render, so it tracks the
    // card as it moves or grows.
    return `M ${start.x} ${start.y} Q ${cx} ${cy} ${end.x} ${end.y}`;
  }

  // Each worker→worker edge's path, computed once and shared by the solid edge
  // layer and the hover/scissors hit layer below (mirrors spokeGeoms).
  const workerEdgeGeoms = workerEdges.map((e) => ({
    from: e.from,
    to: e.to,
    d: workerEdgePath(e.from, e.to),
  }));

  // Judge spoke endpoints (each connected worker's edge → the Judge's edge),
  // computed once and shared by the visible spoke layer and the scissors layer.
  const spokeGeoms = rects.judge
    ? connectedWorkers.flatMap((w) => {
        const wr = rects.workers[w.id];
        if (!wr) return [];
        const jr = rects.judge!;
        const wc = rectCenter(wr);
        const jc = rectCenter(jr);
        return [{
          id: w.id,
          start: rectEdgeToward(wr, jc.x, jc.y),
          end: rectEdgeToward(jr, wc.x, wc.y),
        }];
      })
    : [];

  // Cut worker→Judge spokes (workers the user removed from the Judge). These are
  // NOT erased from the canvas — they stay as a faint dashed "ghost" arrow with a
  // "+" badge at its midpoint, so the relationship reads as "--+-->". Clicking
  // the + restores the link. Same endpoint math as the live spokes; we also keep
  // the midpoint so the + badge can sit on the arrow.
  const cutSpokeGeoms = rects.judge
    ? workers
        .filter((w) => judgeExcluded.has(w.id))
        .flatMap((w) => {
          const wr = rects.workers[w.id];
          if (!wr) return [];
          const jr = rects.judge!;
          const wc = rectCenter(wr);
          const jc = rectCenter(jr);
          const start = rectEdgeToward(wr, jc.x, jc.y);
          const end = rectEdgeToward(jr, wc.x, wc.y);
          return [{
            id: w.id,
            start,
            end,
            mid: { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
          }];
        })
    : [];

  // Cards mirror the Configure Agents pool — with an empty pool there's nothing
  // to show, and there's no on-canvas "add" anymore, so point the user upstream.
  if (steps.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 border border-dashed rounded-lg text-sm text-muted-foreground text-center px-6">
        Add agents in “Configure Agents” above — they appear here automatically.
      </div>
    );
  }

  return (
    <div className="space-y-0">
      {workers.length > 0 && (
        <div className="text-sm font-semibold text-foreground pb-3">
          {meshMode
            ? "Every agent sees every other agent’s output and revises over the configured rounds — the lines show the all-to-all discussion."
            : hasHub
              ? connectingFrom
                ? "Connecting… click another worker to feed it this worker’s output (Esc to cancel)"
                : "Click a worker’s “Connect” button, then another worker to chain them. Click any arrow to cut it; a cut worker→Judge link stays as a faint arrow with a + — click the + to reconnect it."
              : connectingFrom
                ? "Connecting… click another agent to feed it this agent’s output (Esc to cancel)"
                : "Click an agent’s “Connect” button, then another agent to connect them. Click any arrow to remove a connection."}
        </div>
      )}
      <div
        ref={containerRef}
        className="relative mx-auto w-full"
        style={{ maxWidth: CANVAS_MAX_W, height: canvasH, overflow: "visible" }}
      >
        {/* Judge spokes — SLATE-GREY dashed arrows from each CONNECTED worker's
            edge inward to the Judge hub (distinct colour from the black
            worker→worker links). Cut ones (judgeExcluded) are omitted. */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none text-slate-500"
          style={{ overflow: "visible" }}
          aria-hidden
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
          {spokeGeoms.map(({ id, start, end }) => (
            <line
              key={id}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="currentColor"
              strokeWidth="2.5"
              strokeDasharray="6 4"
              markerEnd="url(#reconciler-arrow)"
            />
          ))}
        </svg>

        {/* Faint echo + scissors hit area for the Judge spokes (above cards,
            z-20). Hovering a connected worker's spoke reveals it and shows the
            scissors cursor; clicking cuts that worker→Judge link. Mirrors the
            worker→worker cut layer. */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none z-20 text-slate-500"
          style={{ overflow: "visible" }}
          aria-hidden
        >
          <defs>
            <marker
              id="reconciler-arrow-ghost"
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
          {spokeGeoms.map(({ id, start, end }) => (
            <g key={`cut-judge-${id}`} className="group">
              <line
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke="currentColor"
                strokeWidth="2.5"
                strokeDasharray="6 4"
                markerEnd="url(#reconciler-arrow-ghost)"
                className="opacity-0 transition-opacity group-hover:opacity-100 pointer-events-none"
              />
              <line
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke="transparent"
                strokeWidth="22"
                strokeLinecap="round"
                className={connectingFrom ? "pointer-events-none" : "pointer-events-auto"}
                style={connectingFrom ? undefined : { cursor: SCISSORS_CURSOR }}
                onClick={() => onCutJudge?.(id)}
              >
                <title>Click to cut this worker’s link to the Judge</title>
              </line>
            </g>
          ))}
        </svg>

        {/* Cut worker→Judge spokes — kept on the canvas as a faint slate ghost
            arrow (low opacity) instead of being deleted. The "+" restore badge is
            drawn as HTML below, sitting at the arrow's midpoint so the whole reads
            as "--+-->". Group opacity on the <svg> fades the line AND its
            arrowhead together. */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none text-slate-500 opacity-30"
          style={{ overflow: "visible" }}
          aria-hidden
        >
          <defs>
            <marker
              id="reconciler-arrow-cut"
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
          {cutSpokeGeoms.map(({ id, start, end }) => (
            <line
              key={`cut-ghost-${id}`}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="currentColor"
              strokeWidth="2.5"
              strokeDasharray="6 4"
              markerEnd="url(#reconciler-arrow-cut)"
            />
          ))}
        </svg>

        {/* Worker → worker edges — BLACK dashed curves arcing around the rim
            (distinct colour from the slate-grey Judge spokes). */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none text-foreground"
          style={{ overflow: "visible" }}
          aria-hidden
        >
          <defs>
            <marker
              id="ww-arrow"
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
          {workerEdgeGeoms.map((e, i) => {
            if (!e.d) return null;
            return (
              <path
                key={`${e.from}->${e.to}-${i}`}
                d={e.d}
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeDasharray="6 4"
                markerEnd="url(#ww-arrow)"
              />
            );
          })}
        </svg>

        {/* Faint connection echo + scissors hit area for worker → worker edges.
            Sits ABOVE the cards (z-20) so the clickable ribbon — and the echo it
            reveals on hover — span the arrow's full length, including where it
            runs behind a card. The echo is hidden until hover (when the scissors
            cursor appears) so it never paints over card content unless you are
            about to cut. Mirrors the personalized layout's z-0 (solid, behind
            cards) + z-20 (hover echo + hit area) split. */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none z-20 text-foreground/30"
          style={{ overflow: "visible" }}
          aria-hidden
        >
          <defs>
            <marker
              id="ww-arrow-ghost"
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
          {workerEdgeGeoms.map((e, i) => {
            if (!e.d) return null;
            return (
              <g key={`cut-${e.from}->${e.to}-${i}`} className="group">
                <path
                  d={e.d}
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeDasharray="6 4"
                  fill="none"
                  markerEnd="url(#ww-arrow-ghost)"
                  className="opacity-0 transition-opacity group-hover:opacity-100 pointer-events-none"
                />
                <path
                  d={e.d}
                  stroke="transparent"
                  strokeWidth="22"
                  fill="none"
                  strokeLinecap="round"
                  className={connectingFrom ? "pointer-events-none" : "pointer-events-auto"}
                  style={connectingFrom ? undefined : { cursor: SCISSORS_CURSOR }}
                  onClick={() => onDisconnect?.(e.from, e.to)}
                >
                  <title>Click to remove this connection</title>
                </path>
              </g>
            );
          })}
        </svg>

        {/* Deliberation mesh — a plain dashed line between EVERY pair of cards
            (centre to centre), so the ring reads as "everyone talks to everyone".
            Read-only: there are no directional arrows, no scissors, no Connect. */}
        {meshMode && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none text-primary/40"
            style={{ overflow: "visible" }}
            aria-hidden
          >
            {workers.flatMap((a, i) =>
              workers.slice(i + 1).map((b) => {
                const ra = rects.workers[a.id];
                const rb = rects.workers[b.id];
                if (!ra || !rb) return null;
                const ca = rectCenter(ra);
                const cb = rectCenter(rb);
                return (
                  <line
                    key={`mesh-${a.id}-${b.id}`}
                    x1={ca.x}
                    y1={ca.y}
                    x2={cb.x}
                    y2={cb.y}
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="6 4"
                  />
                );
              }),
            )}
          </svg>
        )}

        {/* Worker cards — placed around the circle */}
        {workers.map((w, i) => {
          const pos = workerPos(i);
          return (
            <div
              key={w.id}
              className="absolute z-10"
              style={{
                left: pos.leftPx,
                top: pos.topPx,
                width: WORKER_W,
                transform: "translate(-50%, -50%)",
              }}
            >
              <div ref={setWorkerRef(w.id)} className="relative w-full">
                <WorkflowStepCard
                  step={w}
                  index={i + 1}
                  label={nodeName(i)}
                  showIndex={false}
                  compact
                  minimal
                  status={statusFor(w.id, stepStatuses)}
                  agents={agents}
                  showInputData
                  inputCols={selectedCols}
                  allCols={allCols}
                  documentInput={documentInput}
                  connectedSources={(w.inputs ?? [])
                    .filter((srcId) => workerIds.has(srcId))
                    .map((srcId) => {
                      const si = workers.findIndex((x) => x.id === srcId);
                      return { id: srcId, label: displayName(workers[si], si) };
                    })}
                  lockAgent
                  boxedName
                  onUpdate={(s) => onUpdate(w.id, s)}
                  onRemove={() => onRemove(w.id)}
                  canRemove={false}
                />

                {/* Connection target — while a connection is in progress, the
                    WHOLE card is the click target (no separate square handle).
                    Clicking anywhere on it completes the connection. Black to
                    match the worker→worker links. (Deliberation has no connecting.) */}
                {allowConnect && connectingFrom && connectingFrom !== w.id && (
                  <button
                    type="button"
                    title={`Click to feed the connecting ${nodeNoun}’s output into this ${nodeNoun}`}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      completeConnect(w.id);
                    }}
                    className="absolute inset-0 z-40 rounded-lg ring-2 ring-foreground bg-foreground/5 hover:bg-foreground/15 cursor-pointer transition-colors"
                  />
                )}

                {/* "Connect" button on top of the card — click to start a
                    worker → worker connection. Hidden on the other cards while
                    connecting, since each of those is then a whole-card target.
                    Deliberation is all-to-all by nature, so it has no Connect. */}
                {allowConnect && (!connectingFrom || connectingFrom === w.id) && (
                  <button
                    type="button"
                    title={`Start a connection from this ${nodeNoun} to another ${nodeNoun}`}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setConnectingFrom((cur) => (cur === w.id ? null : w.id));
                    }}
                    // The `before:` block extends the click target well above and to
                    // the sides of the visible pill (still z-40, above the z-20
                    // scissors ribbons), so a near-miss click reserves the Connect
                    // zone instead of landing on the arrow's cut hit-area underneath.
                    className={`absolute -top-3 left-1/2 -translate-x-1/2 z-40 inline-flex items-center gap-1 rounded-full border-2 border-background px-2.5 py-1 text-[11px] font-semibold shadow-md transition hover:scale-105 cursor-pointer before:absolute before:content-[''] before:-top-10 before:-inset-x-12 before:-bottom-2 ${
                      connectingFrom === w.id
                        ? "bg-foreground text-background ring-2 ring-foreground/40 scale-105"
                        : "bg-foreground/85 text-background hover:bg-foreground"
                    }`}
                  >
                    <Link2 className="h-3.5 w-3.5" strokeWidth={2.75} />
                    {connectingFrom === w.id ? "Connecting…" : "Connect"}
                  </button>
                )}
              </div>
            </div>
          );
        })}

        {/* Judge — the hub at the center. No connection handles (it always
            receives every worker). */}
        {reconciler && (
          <div
            className="absolute z-10"
            style={{
              left: "50%",
              top: judgeTopPx,
              width: JUDGE_W,
              transform: "translate(-50%, -50%)",
            }}
          >
            <div ref={judgeRef} className="relative w-full">
              <WorkflowStepCard
                step={reconciler}
                index={0}
                label="Judge"
                showIndex={false}
                compact
                minimal
                status={statusFor(reconciler.id, stepStatuses)}
                agents={agents}
                showInputData
                inputCols={selectedCols}
                allCols={allCols}
                documentInput={documentInput}
                staticSources={judgeSources}
                lockAgent
                onSwapAgent={(agentId) => onSwapStepAgent?.(reconciler.id, agentId)}
                boxedName
                accent
                onUpdate={(s) => onUpdate(reconciler.id, s)}
                onRemove={() => onRemove(reconciler.id)}
                canRemove={false}
              />
            </div>
          </div>
        )}

        {/* "+" restore badges — one at each cut spoke's midpoint, sitting on the
            faint ghost arrow so it reads as "--+-->". Clicking it re-feeds this
            worker's output to the Judge. Independent of the worker→worker connect
            gesture (the Judge is no longer a click-to-connect target). */}
        {cutSpokeGeoms.map(({ id, mid }) => (
          <button
            key={`restore-judge-${id}`}
            type="button"
            title="Click to reconnect this worker’s output to the Judge"
            onClick={(ev) => {
              ev.stopPropagation();
              onRestoreJudge?.(id);
            }}
            style={{ left: mid.x, top: mid.y, transform: "translate(-50%, -50%)" }}
            className="absolute z-30 h-6 w-6 rounded-full border-2 border-slate-400 bg-background text-slate-500 shadow-sm flex items-center justify-center opacity-60 hover:opacity-100 hover:scale-110 transition cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" strokeWidth={2.75} />
          </button>
        ))}
      </div>
    </div>
  );
}
