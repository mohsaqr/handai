# Learnings

### 2026-02-22 (Tauri polish + export + Manual Coder UX)
- [tauri_webview_download]: WKWebView (macOS system WebView used by Tauri) does not support the HTML `download` attribute on anchor tags. Blob URL + `a.click()` silently does nothing. Fix: detect Tauri via `"__TAURI_INTERNALS__" in window` (Tauri v2; v1 used `"__TAURI__"`) and invoke a Tauri command that calls `tauri_plugin_dialog::blocking_save_file()`.
- [tauri_window_state]: `tauri-plugin-window-state` auto-restores window size/position on plugin init — no manual call needed. `AppHandleExt` only exposes `save_window_state`, not `restore_window_state`.
- [tauri_async_blocking_dialog]: `blocking_save_file()` from tauri-plugin-dialog can be called directly inside an `async fn` Tauri command. It dispatches to the main thread internally; no `spawn_blocking` needed on macOS.
- [tauri_db_path_production]: In Tauri production bundles, the CWD of the sidecar is unpredictable. Always pass `DATABASE_URL` explicitly via `.env("DATABASE_URL", path)` using `app.path().app_data_dir()` → `~/Library/Application Support/{bundle_id}/`.
- [shared_export_util]: Centralising CSV build + download in `src/lib/export.ts` as an async function enables Tauri/browser branching with zero impact on call sites (use `void downloadCSV(...)`). Dynamic import of `@tauri-apps/api/core` inside the Tauri branch avoids bundling it in the browser build.
- [tauri_detect_v2]: Use `"__TAURI_INTERNALS__" in window` to detect Tauri v2 at runtime. `"__TAURI__" in window` is Tauri v1. Both can coexist but v2 is the current standard.
- [settings_deny_list]: Claude Code `settings.json` deny list blocks `git add/commit/push` by default. Removing those entries allows Claude to commit directly; `git merge/rebase/reset` remain blocked for safety.

### 2026-02-22 (web stability + desktop packaging)
- [stability_db_log]: In Next.js API routes, DB log writes (Prisma) that follow a successful LLM call must be wrapped in their own try/catch. Without isolation, a DB failure bubbles to the outer catch and returns 500 to the client even though the LLM result is available — a silent data loss bug.
- [stability_workers]: Use `Promise.allSettled` (not `Promise.all`) for parallel LLM worker arrays. Define a minimum success threshold (≥2). Individual worker failures should produce a partial result, not abort the entire analysis.
- [stability_retry]: `withRetry` should classify non-retryable errors (401, 403, invalid_api_key, authentication) and throw immediately. Auth errors retrying 3× wastes ~3–9s and burns rate limit quota unnecessarily.
- [validation_local]: Zod schemas for LLM routes should use `apiKey: z.string().default("")` not `.min(1)`. Local providers (Ollama, LM Studio) pass an empty string or provider-specific token; enforcing min(1) silently breaks local-only workflows.
- [desktop_standalone]: `output: "standalone"` in `next.config.ts` produces `.next/standalone/server.js` — a self-contained Node.js server. This enables both Electron (uses built-in Node) and Tauri (sidecar) packaging without touching any app code. Has zero effect on web deployment.
- [desktop_electron]: Electron's built-in Node.js (via `process.execPath`) can spawn the standalone Next.js server. No separate Node.js installation needed. Port 3947 avoids clashing with dev server on 3000. Pattern: `spawn(process.execPath, ['server.js'])` → poll port → `loadURL`.
- [desktop_tauri_sidecar]: Tauri Phase A uses `tauri-plugin-shell` to spawn Node.js as a sidecar. Requires a platform-suffixed binary in `src-tauri/binaries/node-{target-triple}`. Bundle size ~85 MB (vs Electron ~160 MB). Phase B migration eliminates the sidecar by moving LLM calls to browser fetch + Prisma to tauri-plugin-sql.
- [hydration_localstorage]: Reading localStorage in `useState` lazy init (with `typeof window !== 'undefined'` guard) causes React hydration mismatches because SSR takes the else branch and renders different HTML. Fix: always init state with safe server default, read localStorage only in `useEffect(() => {}, [])`, guard persist effects with `isMounted` flag.

### 2026-02-16
- [prompt_registry]: System prompts are centrally managed in `core/prompt_registry.py` via `PromptRegistry` class. Each prompt has an ID, name, description, category, module, and default value. The Settings page groups them by category. New modules need explicit registration here to appear in the UI.
- [prompt_registry]: Transform Data, AI Coder, and Model Comparison previously had no registered system prompts. Transform and Model Comparison use user-entered prompts via `st.text_area`; AI Coder builds prompts dynamically via `_build_ai_prompt()`. All three now have defaults registered.
- [session_state]: AI Coder uses `aic_` prefix, Manual Coder uses `mc_` prefix for all session state keys. Changing a key name (e.g. `aic_text_col` to `aic_text_cols`) requires updating init, save, load, config, all rendering, and all processing methods.
- [backward_compat]: When changing session save format (e.g. `text_col` string to `text_cols` list), the load function must handle both old and new formats. Pattern: `text_cols = data.get("text_cols"); if text_cols is None: old_col = data.get("text_col"); text_cols = [old_col] if old_col else []`.
- [streamlit_multiselect]: `st.multiselect` with a `key` parameter stores its value in session state under that key. The `default` parameter only applies on first render; subsequent renders use the session state value. This means changing `default` after first render has no effect.
- [coder_architecture]: Both AI Coder and Manual Coder follow identical patterns: `render_config()` returns `ToolConfig` with `config_data` dict, `render_results()` reads from session state. The immersive dialog and regular view both need updating when changing data access patterns.
- [text_display]: When displaying multiple columns in coders, plain text format uses `[ColName]: value` (for AI prompts), HTML format uses `<b>ColName:</b> value` (for UI display). Single column returns raw value without labels for backward compatibility.
- [testing]: All 488 tests pass via `pytest tests/ -x -q`. Tests are in `tests/` with unit tests under `tests/unit/`. The test suite covers coding tools, settings, automator, codebook, consensus, error classifier, LLM client, model comparison, models, and providers.
- [project_docs]: Project markdown docs live in root (`README.md`, `CONTRIBUTING.md`, `AUTOMATOR_UX_CHANGES.md`, `OPENROUTER_INTEGRATION.md`) and `docs/` (`system-prompts.md`, `sample-data-and-prompts.md`, `CONSENSUS_ENHANCED_JUDGE.md`). When adding features, update `docs/system-prompts.md` and `docs/sample-data-and-prompts.md` for prompt-related changes.
- [run_app]: The app runs via `.venv/bin/streamlit run app.py`. The `run.sh` script doesn't activate the venv, so running `streamlit` directly from shell fails with "command not found". Always use the venv path.

### 2026-02-22 (autosave)
- [crash_recovery]: Dual-slot autosave pattern: write current to AUTOSAVE_KEY, rotate old AUTOSAVE_KEY → AUTOSAVE_PREV_KEY before overwriting. Restore checks primary then falls back to prev. Gives one level of undo against corrupt writes.
- [beforeunload_ref]: `beforeunload` handler must read state via a `useRef` (stateRef) not closures, because it's registered once (`[]` dep array) and can't close over fresh state. A render-sync `useEffect` with no dep array keeps stateRef current after every render.
- [recovery_ux]: Replace toast.info for autosave restore with persistent amber banner (`recovered` state). Banner shows session name, coded count, and relative time. Dismissed by user with ✗. Cleared by `doDataLoaded` when fresh data loads.
- [pending_load_guard]: When `codedCount > 0`, intercept `handleDataLoaded` → set `pendingLoad` state → show Dialog confirmation. Cancel keeps session; "Load anyway" calls `doDataLoaded`. Protects against accidental drag-drop destroying work.
- [autosave_indicator]: `autosaveTime` state + `timeAgo` ticker effect (15s interval) → "✓ Autosaved just now / 15s ago" in session bar. Shows only when `autosaveTime` is set (i.e. data loaded).

### 2026-02-22
- [RunMode pattern]: Replaced boolean `testMode`/`isTestRun` with `RunMode = "preview" | "test" | "full"` across all batch tools. Preview = 3 rows (or 1 doc), Test = 10 rows, Full = all.
- [run_tracking]: `/api/runs` POST creates a run record and returns `{id}`. Lift runId to `useState<string|null>` and set after run completes; render "View in History" link with `ExternalLink` icon.
- [concurrency]: `pLimit` from `p-limit` is already a dependency. Pattern: `const limit = pLimit(concurrency); const tasks = data.map(…limit(…)); await Promise.all(tasks)`.
- [consensus_enhanced]: `enableQualityScoring`/`enableDisagreementAnalysis` checkboxes were UI-only; fixed by passing them in fetch body to API, which does optional extra LLM calls and returns `qualityScores`/`disagreementReason`.
- [codebook_generator]: Changed Stage 3 to use `getPrompt("codebook.definition")` (JSON format) instead of hardcoded Markdown prompt, enabling JSON export. Added Phase A/B split: Stage 1 → pause for theme review → user edits → Stage 2+3.
- [prompts_settings]: `setPromptOverride`/`clearPromptOverride` in `src/lib/prompts.ts` write to localStorage. Settings UI now exposes all 15 prompts in a Collapsible, grouped by category.
- [history_filters]: Status and provider filters are client-side on already-fetched runs. `uniqueProviders` derived from `runs.map(r => r.provider)`. Duration formatted via `new Date(end).getTime() - new Date(start).getTime()`.
- [process_documents]: Old Test button hacked files state; replaced with `processFiles(mode: RunMode)` that computes `targetFiles` without mutating state.
- [ai_coder_batch]: Was sequential for-loop; changed to `pLimit(batchConcurrency)` + `Promise.all`. Abort check goes inside the limit closure.
- [build_verification]: `npm run build` (0 TS errors, 23 routes) + `npm test` (71 tests passing) both pass after all 7 improvements.

### 2026-02-21
- [web_architecture]: Global model selection pattern: `useActiveModel()` hook in `web/src/lib/hooks.ts` returns first enabled+configured provider. All tool pages use this instead of hardcoded `providers["openai"]`. Sidebar footer shows active model + links to Settings.
- [web_manual_coder]: Manual Coder completely rewritten with: 8-color CODE_COLORS palette, color bar strips above buttons, named sessions in localStorage (multiple sessions with save/load/delete), context rows (N rows before/after current, dimmed), auto-advance, standard + one-hot CSV export, session name editing inline.
- [web_ai_coder]: AI Coder upgraded: uses global model via `useActiveModel()`, shows AI suggestions above text with confidence percentages, auto-accepts high-confidence codes (threshold configurable), graceful no-model state with Settings link.
- [web_graceful_nomodel]: All AI tools (transform, automator, qualitative-coder, generate, process-documents, codebook-generator, ai-coder) now check for active model before running and show a friendly amber warning with Settings link instead of throwing errors. Manual Coder has zero AI dependency.
- [web_build]: Build passes 0 TypeScript errors with all 23 routes. AI SDK v6 uses `maxOutputTokens` not `maxTokens`. Zod v4 `z.record()` requires 2 args. `LanguageModelV3` is the current type.

### 2026-02-19
- [coder_session_resume]: The "Load Session" button was trapped inside `render_results()`, which only renders after data is loaded AND "Start Coding" is clicked. Users couldn't resume sessions without first loading a new dataset. Fix: add "Resume Saved Session" section to `render_config()` (visible at top before data upload), save dataframe in session files for uploaded-file sessions, and set `*_coding_started = True` in `_load_session()`.
- [db_migration]: Database path changed from local `./handai_data.db` to `~/Library/Application Support/Handai/handai_data.db`. Old sessions were stranded in the local file. Merged via direct SQLite INSERT from old to new DB.
