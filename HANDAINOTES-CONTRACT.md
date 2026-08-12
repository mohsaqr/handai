# HandaiNotes ↔ Handai Contract

Everything the HandaiNotes notebooks (`handainotes/`) found in, changed from,
or deliberately did differently than Handai — in one place, as a standing
contract. The notebooks were audited against Handai `main` commit
`1f1e7c7402438b0d4cd67af9f1e89e835084107f`; per-capability detail lives in each
package's `handainotes/carm-*/docs/HANDAI_PARITY.md`, and this document is the
actionable index over them.

## How this contract is used

- **When working in Handai**: before touching any area an item names, read that
  item. Every item you touch must end the session in a resolved state — do not
  leave it OPEN if you changed the code it describes.
- **When working in HandaiNotes**: any newly discovered Handai defect or
  deliberate divergence gets **appended here** (new C-number) in the same
  session that discovers it.
- **Statuses**: `OPEN` (no decision yet) → one of
  - `FIXED @<commit>` — the defect was corrected in Handai.
  - `ADOPTED @<commit>` — the notebook design was ported to Handai.
  - `ACKNOWLEDGED — <reason>` — deliberate difference; Handai stays as it is.
    Restoring "parity" against an ACKNOWLEDGED item is a regression, not a fix.

---

## A. Defects found in Handai (proposed: FIX)

### C-1 · Multi-section tabular merge strands CSV headers as data rows
- **Where**: `src/lib/llm-browser.ts` → `documentProcessDirect` (joins section
  outputs with `\n\n---\n\n`), then the joined string is parsed by a line
  splitter in `src/app/process-documents/page.tsx`.
- **What happens**: for a chunked document, every section after the first
  contributes its CSV header row and the `---` separator as if they were data.
- **Notebook fix to mirror**: parse each section to records first, concatenate
  the record lists (`carm-documents/src/documents-core.ts`).
- **Status**: OPEN

### C-2 · Hand-rolled CSV line splitter breaks on quoted fields
- **Where**: `tableResults` in `src/app/process-documents/page.tsx` (~line 661)
  splits model CSV on newlines/commas, so a quoted field containing a comma or
  newline destroys the row.
- **Notebook fix to mirror**: use the real CSV parser — papaparse is already a
  Handai dependency.
- **Status**: OPEN

### C-3 · Fan-out without reduce: every section answers as the whole document
- **Where**: `chunk-text.ts` + `documentProcessDirect` — every document over
  the chunk threshold is always split, and each section independently answers
  the user's instruction.
- **What happens** (observed on real book chapters): "who are the authors?" is
  answered correctly by the title page, wrongly by the running head (×4), and
  wrongly by the bibliography (40 cited authors). Every answer is locally
  correct; the merged result is garbage. A better model does not fix this.
- **Notebook fix to mirror**: reading scope per run — `whole` (one call,
  default; modern models take the 50k-char cap in one call), `combined` (map
  then a synthesis call framed to discard running-head/reference-list answers),
  `sections` (old behavior, for documents that genuinely hold many records).
- **Status**: OPEN

### C-4 · Two extraction paths disagree; one asks the model for CSV
- **Where**: `process-documents` (prompt `document.extraction`) asks for CSV;
  `extract-data` → `documentExtractDirect` asks for a JSON array with hard
  rules. CSV cannot distinguish header from data, cannot say "missing", and
  cannot say "found nothing" — JSON has all three (`[]`, `null`, keys).
- **Notebook fix to mirror**: ask for JSON everywhere; CSV/Excel is an export
  choice made after the run, never a prompt change.
- **Status**: OPEN

### C-5 · Extraction trusts the model's header row / returned keys
- **Where**: process-documents CSV parsing (`header: true` semantics).
- **What happens**: a section that omits its header donates its first data row
  as column names; Papa unions keys across sections and the table grows columns
  like `Modeling_Stability_and_Change_from_a_Complex_Systems_Perspective`.
- **Notebook fix to mirror**: the declared schema is authoritative — parse
  header-less, map positionally, match JSON keys tolerantly.
- **Status**: OPEN

### C-6 · Placeholder strings stored as data
- **Where**: process-documents result handling.
- **What happens**: `N/A`, `unknown`, `(Not Extracted from Document)` are kept
  as cell values; rows of pure placeholders survive as records.
- **Notebook fix to mirror**: placeholders read as empty; a row left entirely
  empty is dropped.
- **Status**: OPEN

### C-7 · An extracted field named `status` collides with the run status column
- **Where**: process-documents result table assembly.
- **Notebook fix to mirror**: reserve `document_name`, `status`, `latency_ms`,
  `note` so an extracted field can never misreport its source or its outcome.
- **Status**: OPEN

### C-8 · Failure reasons are dropped for tabular runs
- **Where**: process-documents — a partially failed document shows a bare
  status with no cause.
- **Notebook fix to mirror**: every non-success row carries *why* (`note`:
  "2 of 7 sections failed: Request timed out"); `unparsed` counted separately
  from `partial`.
- **Status**: OPEN

### C-14 · ai-coder's fallback parser fabricates codes; hallucinations are silently dropped and their confidence redistributed
- **Where**: `parseAIResponse` in `src/app/ai-coder/page.tsx` (~lines 287-345).
- **What happens**: on JSON parse failure the raw text is comma-split and every
  fragment becomes a pseudo-code at confidence 80 — a prose refusal turns into
  codes. Off-codebook codes are dropped with no counter or flag, and because
  renormalization to 100 runs *after* the filter, the dropped codes' probability
  mass inflates the survivors (a row where 90% of mass went to an invented code
  leaves the remaining code at 100%). Labels containing an em/en dash are also
  truncated at the dash (`key.split(/\s*[—–]\s/)[0]`).
- **Notebook fix to mirror**: CarmCoder maps labels onto the codebook (exact →
  normalized → unambiguous containment), records unmatched labels in `ai_note`,
  marks unparseable output `unparsed` with the raw reply kept, and has no
  probabilities to redistribute.
- **Status**: OPEN

### C-15 · `ai_codes` means three different things and comma-splitting breaks labels
- **Where**: `src/app/ai-coder/page.tsx` — batch rows store the raw model JSON
  in `ai_codes`; exports join labels with `"; "`; analytics joins with `", "`;
  consumers split on `,` (`parseCodes` in AnalyticsDialog.tsx:29-34), so
  `"; "` exports mis-split and any label containing a comma breaks all three.
  `onehot` export also writes a column named exactly after each code, silently
  overwriting a same-named input column.
- **Notebook fix to mirror**: one canonical meaning, `ai_*`-namespaced output
  columns that cannot collide with input columns.
- **Status**: OPEN

### C-16 · The "Weighted Kappa" is not kappa, and the agreement metric is circular
- **Where**: `weightedPerCodeKappa` in `src/lib/analytics.ts` (~194-241) is a
  Lin's-CCC-style coefficient labeled "Weighted Kappa" in AnalyticsDialog and
  scored on Landis-Koch bands; degenerate codes return NaN and are silently
  dropped from the macro average. The AI's code set is truncated to top-N where
  N = the human's code count (page.tsx:378-387), so `agreement`,
  precision/recall and the disagreement list are conditioned on the label they
  validate against. Genuinely-Cohen implementations (`perCodeKappa`) exist in
  analytics.ts but are never called by ai-coder.
- **Notebook fix to mirror**: CarmCoder computes per-code Cohen's unweighted
  kappa on presence/absence with undefined kappa reported as "—", never 0.
- **Status**: OPEN

### C-17 · XSS: the current row renders unescaped user data
- **Where**: `src/app/ai-coder/page.tsx` (~773-783) — the current row's text is
  rendered via `dangerouslySetInnerHTML` after `applyAllHighlights`, which does
  not escape the source text (escaping is applied only to non-current rows).
  Any HTML/`<script>` in the dataset executes when its row becomes current.
- **Fix**: escape before highlight-marking, as for the other rows.
- **Status**: OPEN

### C-18 · Dead and decorative code paths in the coding tools
- **Where**: `ReviewPanel.tsx` (197 lines) is never rendered — only its
  `CodeEntry` type is imported; `AICSession.results`/`overrides` are always
  written empty. Prompts `ai_coder.suggestions`, `qualitative.default` and
  `qualitative.rigorous` are registered and user-editable in Settings but never
  used by any page (and `ai_coder.suggestions` asks for a format
  `parseAIResponse` cannot read). `autoAcceptThreshold` is persisted and never
  read. "Start Over" clears sessionStorage but not `aic_autosave`, so the stale
  snapshot restores on next mount.
- **Fix**: delete or wire up; make Settings only offer prompts that do something.
- **Status**: OPEN

### C-19 · qualitative-coder round-trips its codebook through the prompt string
- **Where**: `src/app/qualitative-coder/page.tsx` (~231-282) — session restore
  regex-parses the codebook back out of the saved system-prompt text; the
  paste-CSV import (`extractFromPastedCsv`, ~360-396) splits lines on bare
  commas, so quoted descriptions corrupt. The model contract is a bare
  comma-separated label string stored unvalidated in `ai_code`.
- **Notebook fix to mirror**: the codebook is state (saved/restored as data),
  paste goes through a real CSV parser, and output is validated JSON.
- **Status**: OPEN

### C-20 · extract-data's editable "AI instructions" panel is decorative when fields exist
- **Where**: `src/app/extract-data/page.tsx` builds/edits `aiInstructions` and
  passes it as `systemPrompt`, but `documentExtractDirect`
  (`src/lib/llm-browser.ts:~876`) and the `/api/document-extract` route twin
  replace the system prompt with the hard-coded JSON contract whenever
  `fields.length > 0` — which is the normal case. User edits to the visible,
  editable instructions are silently ignored; only the regex-based session
  restore ever reads them back.
- **Notebook fix to mirror** (CarmExtract): the instructions textarea IS the
  run's system prompt — auto-generated from the cell, editable, and always
  honored; "reset to automatic" restores the generated contract.
- **Status**: OPEN

### C-21 · automator's "AI Instructions" (incl. "Extra Instructions") are never sent to any model
- **Where**: `src/app/automator/page.tsx:205-235` builds the pipeline preamble
  + user extras and passes it to `useBatchProcessor({ systemPrompt })`, which
  forwards it only to `dispatchCreateRun` as run-history metadata
  (`useBatchProcessor.ts:111-119`). The per-step system prompt is hardcoded in
  `automatorRowDirect` / the API route from `step.task` + the field schema —
  the visible, editable Section 3 has zero effect on the model. (Same defect
  class as C-20; also, `prompts.ts` registers `automator.rules` as an editable
  Settings prompt that nothing ever calls.)
- **Notebook fix to mirror** (CarmAutomator): there is no decorative
  instructions box — the step tasks ARE the prompts, and a payload preview
  shows exactly what a step sends.
- **Status**: OPEN

### C-22 · automator: a failed step neither aborts nor surfaces; per-step results are discarded
- **Where**: `src/app/api/automator-row/route.ts:112-114` (and the
  `llm-browser.ts` twin): unparseable step output is skipped, `cumulativeData`
  is left untouched, the chain continues (later steps read `undefined`), and
  the response still reports `success: true`. The `stepResults` array that
  records which steps failed is returned and then entirely discarded by
  `page.tsx:244-255` — the user cannot see that a mid-chain step produced
  nothing. On a terminal row error, even the steps that succeeded are dropped.
- **Notebook fix to mirror** (CarmAutomator): a failed step writes an empty
  column plus a `note` naming the step and reason, the chain continues, the
  unit is `partial`, and Resume re-runs it.
- **Status**: OPEN

### C-23 · automator's silent-failure detector counts only keys absent from the input row
- **Where**: `src/app/automator/page.tsx:256-262` decides "the pipeline
  produced nothing" by `Object.keys(output) − Object.keys(row)`. An output
  field named like an existing input column (e.g. `summary` over a CSV that
  already has `summary`) makes `newKeys` empty on a perfect run → 3 wasted
  full-pipeline retries and a spurious `warning`; conversely one failed step
  among successes reads as clean success.
- **Fix**: track per-step success from `stepResults` (see C-22) instead of
  key-set arithmetic.
- **Status**: OPEN

## B. Notebook designs worth considering in Handai (proposed: ADOPT or ACKNOWLEDGE)

### C-9 · Boot-time local LLM adoption
- Handai probes local servers (`/api/local-models`, sidebar probe) but never
  configures one. The notebooks adopt a detected Ollama/LM Studio at startup —
  **only** into a virgin configuration (untouched default provider, no key),
  guard re-checked after the async probe. Shared pick logic:
  `handainotes/carm-transform/src/transform-core.ts` → `autoLocalChoice`.
- **2026-08-06 — premise changed.** Detection no longer goes through
  `/api/local-models`; all three probe sites now call browser-side
  `probeLocalModels()` (`src/lib/local-provider.ts`), and local providers are
  forced onto the browser-direct path in `shouldUseBrowserDirect()`. On the
  hosted deployment the old server probe reported the *host's* loopback, so
  detection returned `{}` and adoption could never have fired there. The
  sidebar's existing adoption (`useLocalProviderDetection` → auto-enable +
  placeholder model replacement) now actually runs — but it still lacks the
  notebooks' virgin-configuration guard, which is the substance of this item.
- **Status**: OPEN — the adopt-or-acknowledge call on the virgin-config guard is
  still outstanding; only the detection mechanism it rests on has changed.

### C-10 · Run diagnostics: provider-reported token usage, wall-clock throughput
- Notebook `callLLM` returns best-effort token usage (missing = unknown, never
  zero); throughput measured against wall clock, with summed-latency ÷
  wall-clock surfaced as "concurrency achieved". Handai reports neither.
- **Status**: OPEN

### C-11 · Results contain results; diagnostics live elsewhere
- Per-document raw transcripts stapled under the results table made the table
  unreadable (user said so three times). The notebooks moved raw output,
  failures and stats into a collapsed "Run information" section.
- **Status**: OPEN

### C-12 · Open clean; offer the previous session instead of imposing it
- Notebooks boot to a clean page and offer the last session in a banner
  (settings carry over; saved HTML still reopens as itself) —
  `handainotes/NOTEBOOK-SPEC.md` §9.7. Handai's per-tool localStorage autosave
  restores silently. The portability argument (a colleague double-clicks your
  artifact) does **not** apply to a web app in the user's own browser, so this
  may be a legitimate ACKNOWLEDGE for Handai.
- **Status**: OPEN

### C-13 · Full-data viewer
- Notebooks show the loaded data on demand (capped rendering, filter searches
  all rows; documents show the exact extracted text a run sends). Handai tools
  show column pickers/preview rows; whether a full viewer is needed per tool is
  a product decision.
- **Status**: OPEN

### C-24 · Encrypted configurations (instructor-to-class credential sharing)
- 2026-08-12: all eight notebooks can export/import one encrypted `.carmconfig`
  file (AES-256-GCM + PBKDF2 via WebCrypto) carrying provider, model, key,
  model allowlist, pacing and an advisory expiry; import passphrase-unlocks it
  and soft-locks the settings surface (`handainotes/NOTEBOOK-SPEC.md` §9.9,
  shared module `carm-transform/src/shared-config.ts`). Built for classroom
  use: one file + one passphrase instead of distributing N raw keys. Handai
  has no equivalent — its provider config is per-browser localStorage
  (`handai-storage`) with keys entered by hand. Adopting would fit naturally
  as import/export on Handai's Settings page (the crypto module is
  dependency-free and browser-safe; the honest-threat-model rule applies:
  client-side locks are advisory, real limits belong provider-side).
- **Status**: OPEN

## C. Deliberate notebook divergences (pre-resolved: do not "restore parity")

These are **ACKNOWLEDGED** on the notebook side — recorded so nobody undoes
them in either direction. Details: `handainotes/carm-*/docs/HANDAI_PARITY.md`.

- pdf.js runs on the main thread via `globalThis.pdfjsWorker` (a `file://` page
  cannot construct a blob-URL Worker — Handai's strategy cannot be carried
  over, and vice versa).
- DOCX read with zero dependencies (ZIP central directory +
  `DecompressionStream`) instead of mammoth; table rows survive on one line.
- Extracted text is stored inside the saved notebook (re-runs without the
  original PDFs; stated in the privacy notice).
- Duplicate filenames are suffixed (`report (2).pdf`) so results/resume can key
  on the name.
- No background processing across navigation (a notebook is one page).
- Structured fast path omitted in CarmDocuments (CarmTransofrm owns per-row
  tabular work).
- API keys are runtime-memory only; saves abort if a key appears in the
  serialized artifact.
