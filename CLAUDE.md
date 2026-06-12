# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev              # Next.js dev server on :3000 (webpack, not Turbopack)
npm run build            # Standalone production build (server + API routes)
npm run build:static     # Static export for GitHub Pages — runs scripts/build-static.sh
npm start                # Serve the standalone production build
npm test                 # Vitest — all suites
npm run test:watch       # Vitest watch mode
npm run lint             # ESLint
npx tsc --noEmit         # TypeScript strict check (no emit)
npx vitest run src/lib/__tests__/retry.test.ts   # Run a single test file
npx vitest run -t "kappa"                        # Run tests matching a name pattern
```

Postinstall runs `prisma generate`. There is no separate migrate step in dev — the dev SQLite DB is created at `prisma/dev.db` on first run (set `DATABASE_URL=file:./dev.db` if missing).

### Build targets

`next.config.ts` switches `output` based on `STATIC_BUILD=1`:

- **Default (`next build`)** → `output: "standalone"` → `.next/standalone/server.js` plus API routes and Prisma access.
- **`STATIC_BUILD=1` (`build:static`)** → `output: "export"` → `out/` static HTML, no API routes, IndexedDB persistence in the browser.

`scripts/build-static.sh` is required for the static build: it backs up `src/app/api/` and `src/app/history/[id]/page.tsx` to `.tmp_backup/`, removes them, clears `.next/`, runs `STATIC_BUILD=1 NEXT_PUBLIC_STATIC=1 npx next build --webpack`, and restores the backed-up files via a `trap EXIT` handler regardless of success or failure. Both env vars matter: `STATIC_BUILD=1` flips `next.config.ts` to `output: "export"`, while `NEXT_PUBLIC_STATIC=1` is what `llm-dispatch.ts` reads at runtime to set `isStatic`. Do not bypass this script — `output: "export"` is incompatible with API routes and dynamic route segments, and the trap ensures the source tree is always restored.

For GitHub Pages subpath deployments, set `PAGES_BASE_PATH=/repo-name` alongside the static build env vars. `next.config.ts` propagates this to both `basePath` / `assetPrefix` (used by Next's `<Link>`, `<Image>`, and static assets) and to `NEXT_PUBLIC_BASE_PATH` (used by client code that builds raw URLs, e.g. CSS `background-image: url(...)`).

`scripts/build-static.sh` is a bash script and is invoked through `npm run build:static`. On Windows, run it via Git Bash or WSL (the working directory is configured as PowerShell, but bash is available through the harness).

### Deployment (CI)

`.github/workflows/deploy.yml` builds the static export and publishes to GitHub Pages on every push to **`main` or `fix`** (and on `workflow_dispatch`). Note the CI job does **not** call `scripts/build-static.sh` — it inlines its own `mv`-based backup/restore of `src/app/api` and `src/app/history/[id]/page.tsx` (to `/tmp`) around `npx next build --webpack`, setting `STATIC_BUILD=1`, `NEXT_PUBLIC_STATIC=1`, and `PAGES_BASE_PATH=/<repo-name>`. **This logic is duplicated** from the script — if you change which files must be removed for `output: "export"` to succeed, update both the script's `cleanup`/removal block and the workflow's build step, or static deploys will break.

## Architecture

Single codebase, two runtime deployments selected at build time. **The same React pages, the same `getModel()`, the same `withRetry()` run in both modes.** Branching happens in one dispatch layer.

### Tool surface

Each `src/app/<route>/` is a self-contained tool page sharing the dispatch/model/retry stack:

- `transform` — row-by-row prompt application over a dataset (the canonical `useBatchProcessor` tool).
- `qualitative-coder` / `ai-coder` — LLM-assisted qualitative coding; `manual` coding UI also lives here. Per-tool localStorage autosave.
- `abstract-screener` — inclusion/exclusion screening with autosave + one-level undo.
- `automator` — multi-step prompt chains over a dataset.
- `consensus` lives under `model-comparison` — N-rater consensus + kappa (see Consensus pipeline).
- `codebook-generator`, `generate`, `extract-data`, `process-documents` — codebook synthesis, free generation, structured extraction, and PDF/DOCX ingestion.
- `mas-panel` — multi-agent system orchestration (see MAS panel below).
- `history` — run browser; `settings` — provider config + prompt overrides.

### The dispatch layer (`src/lib/llm-dispatch.ts`)

This is the central abstraction. Every tool page calls `dispatchProcessRow`, `dispatchConsensusRow`, `dispatchCreateRun`, `dispatchSaveResults`, etc. The dispatch reads two env vars at module load time:

- `NEXT_PUBLIC_STATIC === "1"` → `isStatic = true` (GitHub Pages build).
- `NEXT_PUBLIC_BROWSER_STORAGE !== "0"` → `useBrowserStorage = true` (default).

When `useBrowserStorage` is true (default in **both** standalone and static builds, for security — keys never traverse the server), LLM calls go through `llm-browser.ts` (direct browser fetch to provider APIs) and run history goes to IndexedDB (`db-indexeddb.ts`). When `NEXT_PUBLIC_BROWSER_STORAGE=0`, dispatch falls back to `/api/*` routes and Prisma+SQLite. This means the standalone build's API routes are only used in self-hosted/trusted-server deployments — and even then, only when the operator opts in.

When adding a new LLM tool, **always add a new dispatch function in `llm-dispatch.ts`** rather than calling `fetch('/api/...')` from a page. The dispatch is what makes the static build work; bypassing it will silently break the GitHub Pages target.

### The model abstraction (`src/lib/ai/providers.ts`)

`getModel(provider, modelId, apiKey, baseUrl?)` returns a Vercel AI SDK `LanguageModelV3`. **Pure fetch only — no Node.js APIs.** This property is load-bearing: the same function runs in API route handlers (Node) and in the browser (static build, WebView). Routes covered: OpenAI, Anthropic, Google, Groq, Together (OpenAI-compat), Azure, OpenRouter (OpenAI-compat with custom headers), Ollama (`localhost:11434/v1`), LM Studio (`localhost:1234/v1`), and `custom` (any OpenAI-compatible base URL).

### Retry policy (`src/lib/retry.ts`)

`withRetry()` does exponential backoff with jitter (3 attempts, base 100ms doubling each retry) but **fast-fails on non-retryable errors**. Classification is structured-first: if the thrown error has a numeric `statusCode` or `status` property, it's checked against `NON_RETRYABLE_CODES` (`400`, `401`, `403`, `404`, `422`). Otherwise it falls back to substring matching the lowercased message against `NON_RETRYABLE_TOKENS` (`401`, `403`, `400`, `invalid_api_key`, `authentication`, `authorization`, `bad request`, `invalid request`). When extending either list, remember: anything matched is never retried, so be conservative.

### State management

- **`src/lib/store.ts`** — Zustand store for provider config (API keys, enabled state, default model, base URLs), persisted to `localStorage["handai-storage"]`. The `persist` middleware's `merge` function always spreads `DEFAULT_PROVIDERS` first, so users picking up a new release see new providers without a data migration.
- **`src/lib/processing-store.ts`** — global Zustand store tracking which tool is actively processing, keyed by `toolId` (the route path, e.g. `/transform`). This is what makes background processing **survive navigation** — the user can leave a tool mid-run and come back to find it still progressing. `useBatchProcessor` writes full state here. Abort flags are kept in a module-level `Map` outside Zustand (non-serializable).
- **`src/hooks/useBatchProcessor.ts`** — the shared batch driver used by Transform, Qualitative Coder, etc. It uses `p-limit` for concurrency control, integrates with `processing-store`, and calls `dispatchSaveResults` to log results. New batch tools should build on this hook rather than re-implementing the loop.
- **Per-tool autosave** (AI Coder, Manual Coder, Abstract Screener) writes session state to `localStorage` keys like `aic_autosave` after every change, with a `_prev` slot for one-level undo and a final sync write on `beforeunload` via a ref (avoids stale closure).
- **Shared tool-page hooks (`src/hooks/`)** compose per-page UI/state so tools don't re-implement it — prefer them over hand-rolling state in a new tool. `useSessionState(key, init)` is `useState` backed by `sessionStorage` (survives tool-to-tool navigation but not a browser close; gates writes on a `hydrated` flag so the mount commit can't clobber stored data — distinct from the `localStorage` autosave above). `useAIInstructions(buildAuto)` keeps an auto-generated instruction block in sync while preserving any user text typed after the `"Extra Instructions (Optional) :"` marker (back-compatible with retired marker strings). `useColumnSelection`, `useFileStatesState` / `useFilesRef`, `usePersistedPrompt`, and `useProcessingFlag` cover column picking, upload state, prompt persistence, and `processing-store` wiring respectively.

**Hydration rule:** all `localStorage` reads happen inside `useEffect`, never in render or `useState` lazy initializers. SSR renders without `localStorage`; client must match on first paint and then update on effect.

### Session restore (`src/lib/restore-store.ts`)

Resuming a past run is a cross-page hand-off, not a query. The History detail page calls `setPending(payload)` (original rows, results, provider/model, system prompt), navigates to the tool route, and the tool calls `consume()` **once** on mount to drain it (one-time read — `consume` clears the pending slot). A tool that supports restore wires this through `useRestoreSession`. The original input rows must be reconstructed from the saved run, and document-mode/unstructured runs collapse per-row data to a `{document_name}` marker — so reconstruction has fallbacks (see the MAS panel restore branch in `page.tsx` for the most involved example, including its `cfg.previewRows` snapshot embedded in the saved WORKFLOW CONFIG).

### Prompts (`src/lib/prompts.ts`)

~29 built-in prompts grouped by category: `transform`, `qualitative`, `consensus`, `codebook`, `generate`, `automator`, `screener`, `ai_coder`, `agents`, `document`. `getPrompt(id)` checks `localStorage["handai_prompt_override:" + id]` first, falling back to the registry default. The Settings page edits these overrides. `getPrompt()` is SSR-safe (guarded by `typeof window`).

### Database layer

Two parallel implementations with the same logical schema:

| Path | Module | Storage |
|---|---|---|
| Server-side (standalone, `BROWSER_STORAGE=0`) | `src/lib/prisma.ts` + `prisma/schema.prisma` | SQLite at `prisma/dev.db` |
| Browser-side (default, both builds) | `src/lib/db-indexeddb.ts` | IndexedDB in the browser |

Schema: `Session` → `Run` → `RunResult` (plus `LogEntry`, `ProviderSetting`, `ConfiguredProvider`, `SystemPromptOverride` in Prisma). The History page and `RunDetailClient` read from whichever backend the dispatch is configured for.

**Prisma client** is a hot-reload-safe singleton (`src/lib/prisma.ts`). Don't `new PrismaClient()` elsewhere.

### API routes (`src/app/api/*`)

Server-side routes for the standalone build. All validate input with Zod schemas from `src/lib/validation.ts` — `apiKey: z.string().default("")` is intentional (local providers like Ollama accept empty keys). Routes: `process-row`, `generate-row`, `consensus-row`, `automator-row`, `document-extract`, `document-analyze`, `document-process`, `agent-network-row`, `local-models`, `runs/`, `results/`. The static build's `scripts/build-static.sh` removes this directory before `next build`.

DB log writes in API routes are wrapped in **isolated** `try/catch` — a Prisma error must never prevent the LLM result from reaching the client.

### Consensus pipeline

`/api/consensus-row` and `consensusRowDirect` implement the same flow: N workers (2–5) run in parallel via `Promise.allSettled`, Cohen's kappa + N×N pairwise agreement matrix are computed (`src/lib/analytics.ts`), a judge model synthesises the workers' outputs, and optional quality scores / disagreement analysis are computed. **Throws only if fewer than 2 workers succeed** (kappa requires at least 2 raters). A single failing worker does not abort the analysis.

### MAS panel (`src/app/mas-panel/`)

Multi-agent orchestration UI. State is split across files that must stay in sync:

- **`workflow-types.ts`** — the model. `WorkflowMode` is one of `reconcilier` (top card synthesises parallel workers — reuses the consensus dispatcher), `sequential` (output of step N feeds N+1), `deliberation` (all agents see each other over rounds), `personalized` (free-form DAG: steps grouped into visual "lines"/`slot`s, data flow defined only by explicit `inputs` edges, executed in `topoOrder`). `migrateLegacyMode` maps the retired `"wizard"` mode to `reconcilier` — preserve this when touching mode strings. `composeStepSystemPrompt` layers per-step task/persona/knowledge **additively on top of** the agent's own prefix (task is primary/first).
- **`agent-library.ts`** (`src/lib/`) — the `Agent` shape and `buildAgentSystemPrefix`. `normalizeAgent()` backfills new fields onto agents loaded from localStorage or imported configs; **call it on every restore path** or older saved agents drive uncontrolled inputs / skip new behavior.
- **`WorkflowLayouts.tsx` / `WorkflowStepCard.tsx`** — per-mode rendering (e.g. SVG edge canvas + click-to-connect for `personalized`). The `personalized`, radial `reconcilier`, and `deliberation` layouts draw their SVG edges/spokes from **measured card rects** (a per-card `ResizeObserver` plus a `signature` over each step's `inputs`/`agentId`), so an arrow tracks its card as the card moves or grows — don't hard-code arrow positions, and keep new size-affecting fields in the signature. The executors live in `page.tsx` (one branch per mode).

**Per-card columns:** a card's Input DATA is `(global selectedCols − step.excludedCols) + step.extraCols`, computed by `includedColumns(step, selectedCols, allCols)` and rendered identically in the card UI — the "+ column" picker offers **every uploaded column** (`allCols`/`previewColumns`), not just the globally-selected ones. The same function feeds each executor branch, so the chips and the run stay in sync.

**Output columns** written by the executors: `worker_N_output` + `judge_output` (reconcilier), `step_N_output` (sequential), label-derived `<label>_output` (personalized), `round_R_<label>_output` (deliberation), each with a `_latency_ms` sibling. The Judge column is `judge_output` (formerly `reconciler_output`); **both names stay in the `FINAL_KEYS` column-ordering sets in `page.tsx` and `RunDetailClient.tsx`** so older saved runs still render. Deliberation feeds each agent its own column subset via an optional per-agent `userContent` on `dispatchAgentNetworkRow` (mirrors the consensus workers' per-worker `userContent`).

A per-agent `maxTokens` overrides the global default. When adding a mode, update `WorkflowMode`, `STEP_MINIMUMS`, the selector, a layout, and a `page.tsx` executor branch together.

### File ingestion (`src/lib/parse-file.ts`)

Uploads split into two kinds, and **this split drives each tool's UI and run logic**:

- **Structured** (`csv`, `xlsx`/`xls`, `json`, `ris`) → `parseStructuredFile` returns `Row[]`. Pages set `hasStructuredFile = true`, render a `ColumnSelector`, and run **per-row** (or **document** mode = whole file as one JSON payload). `parseStructuredBuffer` is the server-side twin (takes a buffer, since API routes receive base64). XLSX with multiple non-empty sheets is concatenated with a synthesized `sheet` column; `.ris` goes through `ris-parser.ts`.
- **Unstructured** (PDF/DOCX/TXT) → `parseStructuredFile` returns `null`, so `hasUnstructuredFile = true`, there are **no columns**, and the file's extracted text becomes the single input. Code that assumes columns must guard on `hasStructuredFile` first — an unstructured file has an empty `selectedCols`.

### Document parsing

For unstructured files, two paths turn PDF/DOCX → text:

- **Server (API route `document-extract`)** — Node-side `pdf-parse` v2 + `mammoth` CommonJS. The route disables the pdfjs worker (`workerSrc = ""`) since it has no DOM.
- **Browser (`src/lib/document-browser.ts`, used by static build)** — `pdfjs-dist` v4 (WASM, `import.meta.url` for worker), `mammoth` browser build.

`next.config.ts` deliberately does **not** list `pdf-parse` / `pdfjs-dist` in `serverExternalPackages` — those packages are ESM-only, and `serverExternalPackages` uses `require()` which fails on ESM. Let the bundler handle them.

## Conventions

- **TypeScript strict** is enforced — `npm run build` requires 0 errors. Path alias `@/*` resolves to `src/*` (configured in `tsconfig.json` and mirrored in `vitest.config.ts`).
- **Tests live under `src/lib/__tests__/`** (Vitest, Node environment). Suites: `analytics`, `prompts`, `retry`, `validation`, `chunk-text`, `generate-ids`.
- **Adding a provider:** extend `DEFAULT_PROVIDERS` in `src/lib/store.ts`, add a case in `getModel()`, and (if it's a local server) wire detection into `/api/local-models` and the AppSidebar probe.
- **Adding a tool page:** mirror an existing tool (e.g. `src/app/transform/`). Use `useBatchProcessor` for row-by-row batching, and add corresponding `dispatch*` functions in `llm-dispatch.ts` covering both the browser-direct path (in `llm-browser.ts`) and, if needed, the server-side path (an API route under `src/app/api/`).
- **Never add `localStorage` reads to render or `useState` initializers** — always `useEffect` (hydration safety).
- **Never call `fetch('/api/...')` directly from a page** — go through `llm-dispatch.ts`, otherwise the static build silently breaks.
- **App version** is a hardcoded literal in the footer of `src/app/layout.tsx` (currently `Handai v2.4`, split across a `<span>` so a bare `Handai v` grep misses it). Release commits are messaged `vX.Y — Last updated <Month Year>`; bump the literal in `layout.tsx` in the same commit so the UI matches the tag.

## Notes on docs in the repo

- `ARCHITECTURE.md` is detailed but partially stale — it describes a Tauri desktop wrapper (Phase B) that is **not currently present in this tree** (no `desktop/tauri/`, no Tauri scripts in `package.json`, no `@tauri-apps/*` deps). The dual-path design it describes is real, but in the current code the two paths are **standalone web** vs **static export to GitHub Pages**, not Tauri. Read `ARCHITECTURE.md` for the LLM call path, consensus protocol, and prompt registry details; ignore the Tauri sections unless they get reintroduced.
- `README.md` is accurate for user-facing usage (tools, providers, install, deployment).
- `BLOG.md` is marketing copy — safe to ignore for engineering work.
