# Handai plugin architecture — a plan

> **Status: proposal, nothing implemented.** Written 2026-09-20, informed by a
> working plugin system built for LAILA. Handai's dual-target build changes the
> answer substantially — read `The constraint that decides everything` first.

## What a plugin would be

Handai is a set of tools: Transform, Generate, Process Documents, Qualitative
Coder, Consensus Coder. Each is a Next.js route with the same skeleton — take
tabular or document input, run rows through an LLM under a prompt, show a
preview, export.

**A plugin is a tool.** Not a theme, not a provider, not a widget. That is the
one extension point worth having, because it is the thing people keep wanting:
a coding scheme specific to a project, a pipeline with a different consensus
rule, an extraction shaped like one lab's data.

A second, much cheaper extension point is worth separating out: **prompt packs**
— a codebook, a schema, a system prompt, shipped as data with no code at all.
Most of what looks like demand for plugins is actually demand for this.

---

## The constraint that decides everything

> "Handai ships as a single codebase that targets two runtime environments with
> **zero application code changes** between them." — `ARCHITECTURE.md` §2

| | Web | Tauri desktop |
|---|---|---|
| Build output | `.next/standalone/server.js` | `out/` — **static export** |
| LLM calls | server → `/api/*` → provider | browser → provider directly |
| DB | Prisma + SQLite | `@tauri-apps/plugin-sql` |
| Size | ~250 MB hosted | **~10 MB**, system WebView |

LAILA's approach — upload a zip, unpack it server-side, `require()` its server
half — **cannot work in Tauri at all.** There is no server. `out/` is static
files. A plugin with a server half would exist on web and vanish on desktop,
which breaks the property the whole architecture is built around.

So the design constraint is: **a plugin must be client-only, or it does not
ship.** That is a restriction and also a simplification — it removes migrations,
in-process loading, capability gating over server resources, and the entire
class of "works on web, missing on desktop" bugs.

---

## Phase 0 — a tool registry (prerequisite)

Today a tool is a route plus a page component, discovered by the router. There
is no value that says "these are the tools". Navigation, the home grid and the
history filter each know the list separately.

```ts
// src/lib/tools/registry.ts
export interface ToolDef {
  id: string;                      // 'qualitative-coder'
  title: string;
  blurb: string;
  icon: string;
  input: 'table' | 'documents' | 'none';
  Page: React.ComponentType<ToolProps>;
  /** Shapes the history row and the export. */
  resultSchema: ResultSchema;
}
```

Built-in tools register through it; the routes become thin wrappers. This is
worth doing on its own merits — it removes three parallel lists — and nothing
below is possible without it.

---

## Phase 1 — prompt packs (data, no code)

The cheap 80%. A pack is JSON:

```json
{
  "id": "org.example.thematic-v2",
  "tool": "qualitative-coder",
  "title": "Thematic coding (Braun & Clarke)",
  "codebook": [ { "code": "...", "definition": "..." } ],
  "systemPrompt": "...",
  "defaults": { "temperature": 0.2, "model": "any" }
}
```

Import from a file, store in the existing DB layer (Prisma on web,
`plugin-sql` on desktop — the abstraction already exists), export to share.
**No code executes.** No security question beyond the prompt content itself,
which the user reads.

This should ship before Phase 2 regardless, and may well end the discussion: if
prompt packs satisfy the demand, a code plugin system is unneeded complexity.

---

## Phase 2 — code tools, bundled at build time

For a tool that needs real logic — a different consensus rule, a custom
parser — the plugin is a module the build includes:

```
handai-plugins/
└── my-tool/
    ├── handai-tool.json    id, title, input, apiVersion
    └── tool.tsx            default-exports a ToolDef
```

`next.config.ts` globs the directory and the registry imports what it finds.
Both targets get the same tools because both are built from the same source —
which is exactly the property §2 protects.

**Cost:** installing a plugin means rebuilding. For desktop that means shipping
a new binary. This is a real limitation and the reason Phase 3 exists.

**What a tool gets:** the existing seams, not new ones —
`llm-dispatch.ts` (`getModel()`, already handles the web/Tauri split),
`withRetry()`, `parse-file.ts`, `export.ts`, `analytics.ts`, the Zustand store.
A plugin that reaches around `llm-dispatch` breaks on one of the two targets;
the API should make the supported path the easy one.

---

## Phase 3 — runtime tools on desktop (optional, decide later)

The only way to add a tool without a rebuild, and it exists only on desktop:
Tauri can read a plugin directory from the user's filesystem, and the WebView
can `import()` it.

This **reintroduces the asymmetry** §2 exists to prevent — desktop could load
tools web cannot. Acceptable only if presented plainly: runtime tools are a
desktop-only power-user feature, and a workspace using them is not portable to
the hosted deployment.

If that trade is unacceptable — and it reasonably might be — Phase 3 should be
dropped and Phase 2 accepted as the ceiling.

---

## Security posture

A code plugin is **trusted code** — it runs in the app with the user's API keys
in `localStorage` (desktop) and their documents in memory. There is no sandbox
that would still let a tool call an LLM and read a file.

Handai's position is materially better than a server-side system's, though, and
the plan should say why rather than hand-wave: a plugin here runs in **one
user's** browser with **their own** keys and **their own** data. It cannot reach
another user, because there is no other user. The blast radius is the person who
installed it.

That makes Phase 2's build-time model genuinely sufficient: if a plugin has to
be in the source tree at build time, the person building it has already read it.

---

## Sequencing, and the decision to make first

1. **Phase 0** — tool registry. Do it; it pays for itself.
2. **Phase 1** — prompt packs. High value, near-zero risk.
3. **Then stop and look.** Does anyone still want something packs cannot do?
4. Only then Phase 2. Phase 3 only if desktop-only asymmetry is acceptable.

The honest framing: **most of the perceived need for plugins here is prompt
configuration, not code.** Building Phase 2 first would be solving the rarer
problem with the more expensive mechanism.

---

## Open questions

- **HandaiNotes.** `HANDAINOTES-CONTRACT.md` already defines a document format.
  If a plugin can produce a HandaiNote, the contract becomes a plugin-facing
  API and needs versioning it may not currently have.
- **History and export.** A plugin's results have to render in `/history` and
  export to CSV. That argues for `resultSchema` being required, not optional —
  otherwise every plugin invents its own row shape.
- **Provider access.** Should a plugin be able to add an LLM *provider*, or only
  consume configured ones? Consuming only is far safer and probably enough.
- **Who is the audience?** If the answer is "the three of us", Phase 0 + Phase 1
  is the whole project and Phases 2–3 should not be built.
