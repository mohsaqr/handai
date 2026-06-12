"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, Plus } from "lucide-react";
import { type Agent } from "@/lib/agent-library";
import {
  type WorkflowMode,
  type WorkflowStep,
  groupLines,
  placeLineSteps,
  buildStepLabels,
  MAX_AGENTS_PER_LINE,
} from "./workflow-types";
import { WorkflowStepCard, type StepStatus } from "./WorkflowStepCard";

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
  /** Reconcilier mode — cut a worker→Judge spoke (scissors). */
  onCutJudge?: (workerId: string) => void;
  /** Reconcilier mode — restore a cut worker→Judge spoke (click worker dot → Judge). */
  onRestoreJudge?: (workerId: string) => void;
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
// pixel value of the CONNECTOR_W Tailwind class (w-72 = 18rem = 288px) so the
// arrows span the gap exactly and touch the cards on both ends. A wide gap keeps
// the step cards compact and gives the connecting arrows room to read clearly.
// (Deliberation reuses CONNECTOR_PX as its column gap, so both modes' cards stay
// the same width.)
const CONNECTOR_W = "w-72";
const CONNECTOR_PX = 288;

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

function SequentialSLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd, selectedCols = [], allCols = [], documentInput }: LayoutProps) {
  // A trailing "+" card always follows the last step in the snake. It occupies
  // the cell at index `addIndex`; clicking it appends a new step (like the empty
  // "+" slots in personalized mode). When the current row is full the "+" wraps
  // to a fresh row with a vertical connector.
  const addIndex = steps.length;
  const cellCount = steps.length + 1; // real steps + the trailing "+" card
  const rows: number[][] = [];
  for (let i = 0; i < cellCount; i += SEQUENTIAL_COLS) {
    const row: number[] = [];
    for (let j = i; j < Math.min(i + SEQUENTIAL_COLS, cellCount); j++) row.push(j);
    rows.push(row);
  }

  // Measure a real step card so the trailing "+" card matches its height when it
  // wraps onto a row of its own (where flex `items-stretch` has nothing taller to
  // stretch against). Only the "+" card uses this — the step cards are untouched.
  const cardRef = React.useRef<HTMLDivElement | null>(null);
  const [cardHeight, setCardHeight] = React.useState<number | null>(null);
  React.useLayoutEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const measure = () => {
      const h = el.getBoundingClientRect().height || null;
      setCardHeight((prev) => (prev === h ? prev : h));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [steps.length]);

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
        const missing = SEQUENTIAL_COLS - row.length;

        return (
          <div key={rowIdx}>
            {/* Row of cards */}
            <div className="flex gap-0 items-stretch">
              {/* Partial odd row: empty slots on the LEFT so cards sit on the right */}
              {missing > 0 && !isEvenRow && emptySlots(missing)}
              {orderedRow.map((globalIdx, colIdx) => {
                const isLastInRow = colIdx === orderedRow.length - 1;
                const isAdd = globalIdx === addIndex;
                return (
                  <React.Fragment key={isAdd ? "add" : steps[globalIdx].id}>
                    <div
                      ref={globalIdx === 0 ? cardRef : undefined}
                      className={`flex-1 min-w-0 ${isAdd ? "flex" : ""}`}
                    >
                      {isAdd ? (
                        <button
                          type="button"
                          onClick={onAdd}
                          title="Add a step"
                          style={cardHeight ? { minHeight: cardHeight } : undefined}
                          className="flex-1 min-h-[7rem] rounded-md border border-dashed border-muted-foreground/40 flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-muted-foreground/70 hover:bg-muted/40 transition-colors"
                        >
                          <Plus className="h-5 w-5" />
                        </button>
                      ) : (
                        <WorkflowStepCard
                          step={steps[globalIdx]}
                          index={globalIdx}
                          label={`Step ${globalIdx + 1}`}
                          status={statusFor(steps[globalIdx].id, stepStatuses)}
                          agents={agents}
                          compact
                          showInputData
                          inputCols={selectedCols}
                          allCols={allCols}
                          documentInput={documentInput}
                          prevStepLabel={globalIdx > 0 ? `Step ${globalIdx}` : null}
                          onUpdate={(s) => onUpdate(steps[globalIdx].id, s)}
                          onRemove={() => onRemove(steps[globalIdx].id)}
                          canRemove={steps.length > 2}
                        />
                      )}
                    </div>
                    {!isLastInRow && (
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
    </div>
  );
}

// ── Reconcilier hierarchy: workers on top, reconciler below with lines down ──

function ReconcilierHierarchyLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd, onConnect, onDisconnect, onCutJudge, onRestoreJudge, selectedCols = [], allCols = [], documentInput }: LayoutProps) {
  const reconciler = steps[0];
  const workers = steps.slice(1);
  const workerIds = new Set(workers.map((w) => w.id));

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
        : connectedWorkers.map((w) => `Worker ${workers.indexOf(w) + 1} output`);

  // Measure the real card positions so each spoke can run from its worker's
  // bottom edge straight to the Judge's top edge. A fixed-height band with
  // percentage x-positions breaks once workers wrap onto a second row — the
  // lines start floating below the cards instead of touching them. Measuring
  // the actual DOM rects (like PersonalizedLayout) keeps the arrows anchored to
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
  }, [signature, resizeTick]);

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
  // the Judge→worker spokes are as long as possible. Card widths are FIXED px.
  const WORKER_W = 400;
  const JUDGE_W = 410;
  const CANVAS_MAX_W = 1700;
  const CANVAS_MAX_H = 720;
  const N = workers.length;
  // Orientation: pointy-top by default (triangle/pentagon point up). Two workers
  // flank the Judge left/right; four sit at the corners of an upright square.
  const startDeg = N === 2 ? 180 : N === 4 ? -45 : -90;
  const angles = Array.from(
    { length: N },
    (_, i) => ((startDeg + (360 / Math.max(N, 1)) * i) * Math.PI) / 180,
  );

  // Canvas = the positioning box. Width is the measured container width; height
  // is capped so the diagram doesn't get absurdly tall on a wide panel. Two or
  // fewer workers form a horizontal line (no vertical spread), so a short box
  // avoids a big empty band above and below.
  const canvasW = boxSize;
  // A lone worker stacks vertically above the Judge and needs real vertical room
  // for a visible spoke. Two workers flank the Judge horizontally (no vertical
  // spread), so a short box avoids empty bands above and below.
  const isVerticalStack = N === 1;
  const canvasH = isVerticalStack
    ? Math.min(boxSize, 500)
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
  const workerPos = (i: number) => {
    // Single worker: pin it near the top, centered over the Judge, so the spoke
    // runs straight down. The radial formula would place it only ~ry above the
    // centered Judge — too little to clear the Judge's own height, so the cards
    // would overlap with no room for the arrow.
    if (isVerticalStack) {
      return { leftPx: canvasW / 2, topPx: MARGIN + CARD_HALF_H };
    }
    return {
      leftPx: canvasW / 2 + rx * Math.cos(angles[i]),
      topPx: canvasH / 2 + ry * Math.sin(angles[i]),
    };
  };
  // Judge sits at the canvas center for the radial layouts; for the lone-worker
  // vertical stack it drops to the bottom so the worker→Judge spoke spans the
  // whole canvas instead of being squeezed into one half.
  const judgeTopPx = isVerticalStack ? canvasH - MARGIN - CARD_HALF_H : canvasH / 2;

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

  // Worker → worker edge: source edge → target edge, bowed OUTWARD (away from
  // the Judge hub) so it arcs around the rim instead of crossing the center.
  function workerEdgePath(from: string, to: string): string | null {
    const a = rects.workers[from];
    const b = rects.workers[to];
    if (!a || !b) return null;
    const ac = rectCenter(a);
    const bc = rectCenter(b);
    const start = rectEdgeToward(a, bc.x, bc.y);
    const end = rectEdgeToward(b, ac.x, ac.y);
    const hub = rects.judge ? rectCenter(rects.judge) : { x: (ac.x + bc.x) / 2, y: (ac.y + bc.y) / 2 };
    const mx = (start.x + end.x) / 2;
    const my = (start.y + end.y) / 2;
    let ox = mx - hub.x;
    let oy = my - hub.y;
    const olen = Math.hypot(ox, oy) || 1;
    ox /= olen;
    oy /= olen;
    const bow = 48;
    const cx = mx + ox * bow;
    const cy = my + oy * bow;
    // End exactly on the target card's edge so the arrow reaches the card. The
    // endpoint is recomputed from the live rect each render, so it tracks the
    // card as it moves or grows.
    return `M ${start.x} ${start.y} Q ${cx} ${cy} ${end.x} ${end.y}`;
  }

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

  return (
    <div className="space-y-0">
      {workers.length > 0 && (
        <div className="text-sm font-semibold text-foreground pb-3">
          {connectingFrom
            ? "Connecting… click another worker to feed it this worker’s output, or the Judge to reconnect it (Esc to cancel)"
            : "Click a worker’s ↗ output handle, then another worker to chain them — or the Judge to reconnect a cut link. Click any arrow to cut it (including a worker’s link to the Judge)."}
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
          {workerEdges.map((e, i) => {
            const d = workerEdgePath(e.from, e.to);
            if (!d) return null;
            return (
              <path
                key={`${e.from}->${e.to}-${i}`}
                d={d}
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
          {workerEdges.map((e, i) => {
            const d = workerEdgePath(e.from, e.to);
            if (!d) return null;
            return (
              <g key={`cut-${e.from}->${e.to}-${i}`} className="group">
                <path
                  d={d}
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeDasharray="6 4"
                  fill="none"
                  markerEnd="url(#ww-arrow-ghost)"
                  className="opacity-0 transition-opacity group-hover:opacity-100 pointer-events-none"
                />
                <path
                  d={d}
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

        {/* Worker cards — placed around the circle */}
        {workers.map((w, i) => {
          const pos = workerPos(i);
          // Put the output dot on the OUTER side (away from the Judge): cards to
          // the right of centre get it on their RIGHT, cards to the left on their
          // LEFT. The inner, Judge-facing edge is where the permanent spokes and
          // most incoming worker→worker arrowheads land, so keeping the dot on
          // the outer side stops the dot from coinciding with an arrowhead.
          const onRight = pos.leftPx > canvasW / 2;
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
                  label={`Worker ${i + 1}`}
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
                    .map((srcId) => ({
                      id: srcId,
                      label: `Worker ${workers.findIndex((x) => x.id === srcId) + 1}`,
                    }))}
                  onUpdate={(s) => onUpdate(w.id, s)}
                  onRemove={() => onRemove(w.id)}
                  canRemove={steps.length > 2}
                />

                {/* Connection target — while a connection is in progress, the
                    WHOLE card is the click target (no separate square handle).
                    Clicking anywhere on it completes the connection. Black to
                    match the worker→worker links. */}
                {connectingFrom && connectingFrom !== w.id && (
                  <button
                    type="button"
                    title="Click to feed the connecting worker’s output into this worker"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      completeConnect(w.id);
                    }}
                    className="absolute inset-0 z-40 rounded-lg ring-2 ring-foreground bg-foreground/5 hover:bg-foreground/15 cursor-pointer transition-colors"
                  />
                )}

                {/* Output point (outer edge) — click to start a worker → worker
                    connection. Hidden on the other cards while connecting, since
                    each of those is then a whole-card target. */}
                {(!connectingFrom || connectingFrom === w.id) && (
                  <button
                    type="button"
                    title="Start a connection from this worker to another worker"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setConnectingFrom((cur) => (cur === w.id ? null : w.id));
                    }}
                    className={`absolute ${onRight ? "-right-3" : "-left-3"} top-1/2 -translate-y-1/2 z-40 h-7 w-7 rounded-full border-2 border-background shadow-md flex items-center justify-center transition hover:scale-110 cursor-pointer ${
                      connectingFrom === w.id
                        ? "bg-foreground ring-2 ring-foreground/40 scale-110"
                        : "bg-foreground/85 hover:bg-foreground"
                    }`}
                  >
                    <ArrowUpRight className="h-4 w-4 text-background" strokeWidth={2.75} />
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
                onUpdate={(s) => onUpdate(reconciler.id, s)}
                onRemove={() => onRemove(reconciler.id)}
                canRemove={steps.length > 2}
              />

              {/* Reconnect target — while connecting from a worker whose Judge
                  link was cut, the whole Judge card is a click target that
                  restores the slate worker→Judge spoke. */}
              {connectingFrom && judgeExcluded.has(connectingFrom) && (
                <button
                  type="button"
                  title="Click to reconnect this worker’s output to the Judge"
                  onClick={(ev) => {
                    ev.stopPropagation();
                    onRestoreJudge?.(connectingFrom);
                    setConnectingFrom(null);
                  }}
                  className="absolute inset-0 z-40 rounded-lg ring-2 ring-slate-500 bg-slate-500/5 hover:bg-slate-500/15 cursor-pointer transition-colors"
                />
              )}
            </div>
          </div>
        )}
      </div>

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

function DeliberationNetworkLayout({ steps, agents, stepStatuses, onUpdate, onRemove, onAdd, selectedCols = [], allCols = [], documentInput }: LayoutProps) {
  // Fixed 3 columns; rows fill left-to-right. Cards fill their column (1fr) and
  // the column gap matches the Sequential connector width (CONNECTOR_PX), so the
  // participant cards come out the same size as the Sequential step cards.
  const cols = 3;

  // The peer-to-peer mesh connects every pair of cards. Draw it from the cards'
  // MEASURED centres rather than fixed percentages: with the wide column gaps
  // the grid cell centres no longer sit at simple (col+0.5)/cols fractions, so a
  // percentage mesh would float off the cards. Measuring locks every line to the
  // real card centres as the grid wraps or a card resizes.
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const cardRefs = React.useRef<Map<string, HTMLDivElement>>(new Map());
  const [centers, setCenters] = React.useState<Record<string, { x: number; y: number }>>({});
  const [resizeTick, setResizeTick] = React.useState(0);

  const signature = steps.map((s) => `${s.id}:${s.agentId ?? ""}`).join(",") + `#${steps.length}`;

  React.useEffect(() => {
    const ro = new ResizeObserver(() => setResizeTick((t) => t + 1));
    if (containerRef.current) ro.observe(containerRef.current);
    cardRefs.current.forEach((el) => ro.observe(el));
    const onWin = () => setResizeTick((t) => t + 1);
    window.addEventListener("resize", onWin);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", onWin);
    };
  }, [signature]);

  React.useLayoutEffect(() => {
    const c = containerRef.current;
    if (!c) return;
    const cr = c.getBoundingClientRect();
    const next: Record<string, { x: number; y: number }> = {};
    cardRefs.current.forEach((el, id) => {
      const r = el.getBoundingClientRect();
      next[id] = { x: r.left - cr.left + r.width / 2, y: r.top - cr.top + r.height / 2 };
    });
    setCenters((prev) => {
      const keys = Object.keys(next);
      const same =
        keys.length === Object.keys(prev).length &&
        keys.every((k) => prev[k] && prev[k].x === next[k].x && prev[k].y === next[k].y);
      return same ? prev : next;
    });
  }, [signature, resizeTick]);

  const setCardRef = (id: string) => (el: HTMLDivElement | null) => {
    if (el) cardRefs.current.set(id, el);
    else cardRefs.current.delete(id);
  };

  return (
    <div className="space-y-3">
      <div ref={containerRef} className="relative">
        {/* Peer-to-peer mesh — dashed connector between every pair of agents,
            anchored to the measured card centres. */}
        {steps.length >= 2 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none text-primary/40"
            style={{ overflow: "visible" }}
            aria-hidden
          >
            {steps.flatMap((s, i) =>
              steps.slice(i + 1).map((s2) => {
                const a = centers[s.id];
                const b = centers[s2.id];
                if (!a || !b) return null;
                return (
                  <line
                    key={`${s.id}-${s2.id}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke="currentColor"
                    strokeWidth="2.75"
                    strokeDasharray="6 4"
                  />
                );
              })
            )}
          </svg>
        )}

        {/* Card grid — `relative` keeps the cards painting on top of the absolute
            mesh SVG. Cards fill their 1fr column; the 208px column gap matches the
            Sequential connector width so the cards come out the same size. */}
        <div
          className="grid gap-y-24 relative"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, columnGap: CONNECTOR_PX }}
        >
          {steps.map((step, i) => (
            <div key={step.id} ref={setCardRef(step.id)}>
              <WorkflowStepCard
                step={step}
                index={i}
                label={`Participant ${i + 1}`}
                showIndex={false}
                status={statusFor(step.id, stepStatuses)}
                compact
                agents={agents}
                showInputData
                inputCols={selectedCols}
                allCols={allCols}
                documentInput={documentInput}
                onUpdate={(s) => onUpdate(step.id, s)}
                onRemove={() => onRemove(step.id)}
                canRemove={steps.length > 2}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="pt-4">
        <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={onAdd}>
          <Plus className="h-3.5 w-3.5" /> Add participant
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
  allCols = [],
  documentInput,
  lineCount = 2,
}: LayoutProps) {
  // Lines are visual rows only — data flow is the explicit `inputs` edges.
  // Render every line from 0..maxLine (not just lines that still have a card),
  // so a line whose last card was cleared stays visible as three empty "+" slots
  // instead of vanishing. Line numbers come from the steps' actual `line` value.
  const grouped = new Map(groupLines(steps));
  const maxStepLine = steps.length > 0 ? Math.max(...steps.map((s) => s.line ?? 0)) : -1;
  // Base line span: at least `lineCount` lines (what "Add AI Agent line" controls),
  // always enough to cover every populated line, and floored at 2 lines total so
  // the canvas never collapses to the "no agents yet" prompt.
  const baseMax = Math.max(1, lineCount - 1, maxStepLine);
  // If every slot across those lines is full, append one fresh empty line so the
  // user always has somewhere to add the next agent — filling the last "+" reveals
  // a new line automatically.
  let allFull = true;
  for (let ln = 0; ln <= baseMax; ln++) {
    if ((grouped.get(ln)?.length ?? 0) < MAX_AGENTS_PER_LINE) { allFull = false; break; }
  }
  const maxLine = allFull ? baseMax + 1 : baseMax;
  const lines: [number, WorkflowStep[]][] = [];
  for (let ln = 0; ln <= maxLine; ln++) lines.push([ln, grouped.get(ln) ?? []]);
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

  // Match every empty "+" slot to the height of the real agent cards (only those
  // are measured into `rects`). Without this, a row that holds no card — e.g. the
  // default empty second line — would shrink its "+" boxes to min-h-[7rem] instead
  // of matching the taller card-sized "+" that sits next to actual agents.
  const measuredCardHeights = Object.values(rects)
    .map((r) => r.h)
    .filter((h) => h > 0);
  const cardHeight = measuredCardHeights.length ? Math.max(...measuredCardHeights) : null;

  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-foreground">
        {connectingFrom
          ? "Connecting… click a target agent (Esc to cancel)"
          : "Click an agent’s ↗ output handle, then click another agent to connect. Click an arrow to remove it."}
      </div>

      {lines.length === 0 ? (
        <div className="flex items-center justify-center py-12 border border-dashed rounded-lg text-sm text-muted-foreground">
          No agents yet — click “+ Add AI Agent line” to start a line.
        </div>
      ) : (
        <div ref={canvasRef} className="relative space-y-6">
          {/* Solid connection line. Sits BEHIND the cards (z-0) so the full-
              strength arrow shows in the open space between cards without
              painting over card content. Purely visual — the clickable scissors
              ribbon lives on the faint top layer (z-20) below, so it spans the
              line's full length, including where it runs behind a card. */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-0"
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
                </g>
              );
            })}
          </svg>

          {lines.map(([lineNo, lineSteps]) => {
            const placed = placeLineSteps(lineSteps);
            return (
              <div key={lineNo} className="relative z-10">
                <div className="flex items-stretch">
                  {placed.map((step, slot) => (
                    <div
                      key={step?.id ?? `slot-${slot}`}
                      className="flex-1 min-w-0 flex px-28"
                    >
                      {step ? (
                        <div ref={setCardRef(step.id)} className="relative w-full">
                          <WorkflowStepCard
                            step={step}
                            index={slot}
                            label={`Agent ${lineNo * MAX_AGENTS_PER_LINE + slot + 1}`}
                            showIndex={false}
                            compact
                            status={statusFor(step.id, stepStatuses)}
                            agents={agents}
                            showInputData
                            inputCols={selectedCols}
                            allCols={allCols}
                            documentInput={documentInput}
                            connectedSources={(step.inputs ?? []).map((srcId) => ({
                              id: srcId,
                              label: stepLabels[srcId] ?? srcId,
                            }))}
                            onUpdate={(s) => onUpdate(step.id, s)}
                            onRemove={() => onRemove(step.id)}
                            canRemove
                          />

                          {/* Input target — while a connection is in progress,
                              the WHOLE card is the click target (no separate
                              square handle), matching the Judge layout. Clicking
                              anywhere on it completes the connection. */}
                          {connectingFrom && connectingFrom !== step.id && (
                            <button
                              type="button"
                              title="Click to feed the connecting agent's output into this agent"
                              onClick={(ev) => {
                                ev.stopPropagation();
                                completeConnect(step.id);
                              }}
                              className="absolute inset-0 z-40 rounded-lg ring-2 ring-primary bg-primary/5 hover:bg-primary/15 cursor-pointer transition-colors"
                            />
                          )}

                          {/* Output handle (right) — click to start a connection.
                              Hidden on the other cards while connecting, since each
                              of those is then a whole-card target. */}
                          {(!connectingFrom || connectingFrom === step.id) && (
                            <button
                              type="button"
                              title="Start a connection from this agent"
                              onClick={(ev) => {
                                ev.stopPropagation();
                                setConnectingFrom((cur) =>
                                  cur === step.id ? null : step.id,
                                );
                              }}
                              className={`absolute -right-3 top-1/2 -translate-y-1/2 z-40 h-7 w-7 rounded-full border-2 border-background shadow-md flex items-center justify-center transition hover:scale-110 cursor-pointer ${
                                connectingFrom === step.id
                                  ? "bg-primary ring-2 ring-primary/40 scale-110"
                                  : "bg-primary/85 hover:bg-primary"
                              }`}
                            >
                              <ArrowUpRight className="h-4 w-4 text-primary-foreground" strokeWidth={2.75} />
                            </button>
                          )}
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onAddToLine?.(lineNo, slot)}
                          title="Add an agent here"
                          style={cardHeight ? { minHeight: cardHeight } : undefined}
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

          {/* Faint connection echo + scissors hit area. Sits ABOVE the cards
              (z-20) so the clickable ribbon is exposed along the arrow's full
              length, including where it runs behind a card. The visible echo is
              hidden by default and only fades in on hover (when the scissors
              cursor appears) — so it never paints over card content unless you
              are about to cut it. The <svg> itself is pointer-events-none; only
              the thin transparent ribbon per edge is clickable, so card content
              stays interactive everywhere else. */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-20 text-primary/30"
            style={{ overflow: "visible" }}
            aria-hidden
          >
            <defs>
              <marker
                id="pz-arrow-ghost"
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
                <g key={`ghost-${e.from}->${e.to}-${i}`} className="group">
                  <path
                    d={d}
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="6 4"
                    fill="none"
                    markerEnd="url(#pz-arrow-ghost)"
                    className="opacity-0 transition-opacity group-hover:opacity-100 pointer-events-none"
                  />
                  {/* Scissors hit area — disabled while making a connection so
                      the scissors cursor doesn't show on the way and the ribbon
                      can't steal the target click. Re-enabled once idle. */}
                  <path
                    d={d}
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
        </div>
      )}

      <div className="pt-4">
        <Button
          variant="outline"
          size="sm"
          className="text-xs gap-1.5"
          onClick={() => onAddLine?.()}
        >
          <Plus className="h-3.5 w-3.5" /> Add AI Agent line
        </Button>
      </div>
    </div>
  );
}
