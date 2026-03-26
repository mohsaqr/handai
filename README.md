# Handai AI Data Suite

A qualitative and quantitative data analysis suite powered by large language models. Runs as a Next.js web app with an optional static export for GitHub Pages.

Built with Next.js 16, React 19, TypeScript, and Tailwind CSS v4.

---

## Table of Contents

- [What is Handai?](#what-is-handai)
- [Tools](#tools)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [First Run Walkthrough](#first-run-walkthrough)
- [Configuration and API Keys](#configuration-and-api-keys)
- [Supported LLM Providers](#supported-llm-providers)
- [Data Format](#data-format)
- [Scripts Reference](#scripts-reference)
- [Web Deployment](#web-deployment)
- [Tech Stack](#tech-stack)
- [Project Layout](#project-layout)
- [Architecture](#architecture)

---

## What is Handai?

Handai is a browser-based research toolkit for analysts, qualitative researchers, and data scientists who want to apply LLMs to their data without writing code. You bring a CSV file, pick a tool, choose a model, and run. Results are saved to a local SQLite database and can be exported as CSV at any time.

**Key design principles:**

- **No vendor lock-in.** Switch between OpenAI, Anthropic, Google, Groq, or a locally running Ollama model by changing a dropdown in Settings.
- **Human in the loop.** Tools like AI Coder are built around review workflows, not fire-and-forget automation.
- **Works offline.** When connected to Ollama or LM Studio, the entire app runs without any internet connection.
- **Session persistence.** Navigate between tools freely — active processing continues in the background. Restore any past session from History and resume where you left off.

---

## Tools

| Tool | What it does |
|---|---|
| **AI Coder** | AI suggests a code for each row; you review and accept, override, or skip. Autosave keeps your progress across sessions. |
| **Qualitative Coder** | Batch-code an entire dataset against a codebook in one run. Each row is scored against every codebook category. Exports results as CSV. |
| **Consensus Coder** | N independent worker models each code every row, then a judge model resolves disagreements. Reports Cohen's kappa inter-rater agreement score. |
| **Codebook Generator** | Feed a sample of text rows and let the LLM inductively derive a codebook. Edit and refine before using it in Qualitative Coder. |
| **Abstract Screener** | Screen research abstracts against inclusion/exclusion criteria with AI pre-screening and human review. |
| **Transform** | Apply any free-form LLM prompt to every row in a CSV. Useful for translation, summarisation, sentiment, entity extraction, or any custom transformation. |
| **Automator** | Build a multi-step LLM pipeline where each step's output feeds into the next. Chain up to N steps, each with its own prompt and model. |
| **Generate** | Synthesise realistic datasets by providing a schema and a few example rows. Useful for testing or creating training data. |
| **Process Documents** | Upload PDFs, DOCX, or plain-text files. The LLM extracts structured data fields you define. Results exportable as CSV. |
| **Model Comparison** | Run the same prompt across N models side-by-side. Useful for evaluating model suitability before committing to a full batch run. |
| **History** | Browse all previous batch runs. Filter by tool or date, drill into per-row results, restore any session, and re-export as CSV. |
| **Settings** | Configure API keys for each provider, enable or disable providers, and customise the system prompt templates used by each tool. |

---

## Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Node.js | 20 or higher | `node --version` |
| npm | 10 or higher (ships with Node 20) | `npm --version` |
| Git | Any recent version | `git --version` |

For **local LLM inference**: [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) installed and running. The app auto-detects them — no configuration required.

---

## Installation

```bash
git clone https://github.com/mohsaqr/handai.git
cd handai
npm install
```

That is all that is required for local development. No database setup, no environment variables — the app creates its SQLite database automatically on first run.

---

## First Run Walkthrough

This takes about 5 minutes from a fresh clone to running analysis on your own data.

**1. Start the development server**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

**2. Add an API key**

Click **Settings** in the left sidebar. Select a provider (for example, OpenAI), paste your API key into the key field, and click **Save**. The status indicator next to the provider turns green.

If you prefer not to use a cloud provider, skip this step and set up Ollama instead (see step 3).

**3. (Optional) Use a local model**

Install [Ollama](https://ollama.com) and run any model:

```bash
ollama run llama3.2
```

Handai detects Ollama automatically when you open the app. No key or configuration needed. The detected model appears as a clickable pill in Settings under the Ollama section.

LM Studio works the same way — start the local server in LM Studio and Handai will find it.

**4. Try a tool with sample data**

Click any tool in the sidebar (for example, **Qualitative Coder**). Click **Load Sample Data** to load a built-in demo dataset. Configure your options and click **Run**. Results appear inline and can be exported as CSV.

**5. Use your own data**

Click the file upload area on any tool page and select a CSV file, or drag and drop it. The tool will show a column selector — pick which column contains the text you want to analyse. Column names and order do not matter.

---

## Configuration and API Keys

API keys are stored in your browser's `localStorage` via the Settings page. They never leave your browser except when making direct calls to provider APIs.

No `.env` file is required for local development.

If you are deploying the web app to a server and want to pre-configure keys at the server level (so users do not need to enter them), create `.env.local`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GENERATIVE_AI_API_KEY=...
GROQ_API_KEY=...
TOGETHER_AI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENROUTER_API_KEY=...
```

Keys set in `.env.local` act as server-side defaults. Keys entered in the Settings UI take precedence over server-side keys.

---

## Supported LLM Providers

| Provider | Type | Auto-detected | API Key Required |
|---|---|---|---|
| OpenAI | Cloud | No | Yes |
| Anthropic (Claude) | Cloud | No | Yes |
| Google Gemini | Cloud | No | Yes |
| Groq | Cloud | No | Yes |
| Together AI | Cloud | No | Yes |
| Azure OpenAI | Cloud | No | Yes + endpoint |
| OpenRouter | Cloud | No | Yes |
| Ollama | Local | Yes (port 11434) | No |
| LM Studio | Local | Yes (port 1234) | No |
| Custom endpoint | Local/self-hosted | No | Optional |

The **Custom endpoint** option accepts any OpenAI-compatible API (vLLM, text-generation-webui, LocalAI, etc.). Enter the base URL in Settings.

---

## Data Format

Handai accepts any well-formed CSV file:

- Any delimiter (comma, semicolon, tab — detected automatically)
- Any number of columns
- Column names in the first row (required)
- UTF-8 encoding recommended

After uploading, every tool shows a **column selector** dropdown. Pick the column that contains the text you want to process. You can also select multiple columns for tools that support multi-column input (e.g. Automator).

There is no required schema. A CSV with a single column of free-text responses works just as well as a structured dataset with dozens of columns.

---

## Scripts Reference

```bash
npm run dev          # Start Next.js dev server at http://localhost:3000 with hot reload
npm run build        # Production build — standalone output — 0 TypeScript errors required
npm start            # Serve the production build (run npm run build first)
npm test             # Run Vitest test suite — 115 tests across 4 suites
npm run lint         # ESLint check across all source files
npx tsc --noEmit     # TypeScript type-check (strict mode)
```

### Build targets

| Mode | Command | Output | Used for |
|---|---|---|---|
| Standalone | `npm run build` | `.next/standalone/server.js` | Web deployment, Docker |
| Static export | `npm run build:static` | `out/` | GitHub Pages |

---

## Web Deployment

### Option 1: Node.js server

```bash
npm run build
npm start            # Serves on port 3000
```

Set the `PORT` environment variable to change the port.

### Option 2: Docker

```dockerfile
FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
ENV PORT=3000 HOSTNAME=0.0.0.0
EXPOSE 3000
CMD ["node", "server.js"]
```

Build and run:

```bash
docker build -t handai .
docker run -p 3000:3000 handai
```

### Option 3: GitHub Pages

```bash
npm run build:static
# Deploy the out/ directory to GitHub Pages
```

The static build uses IndexedDB for persistence instead of server-side SQLite.

### Option 4: Managed platforms

Handai is a standard Next.js application. Deploy to Vercel, Railway, Fly.io, or any other platform that supports Node.js without any special configuration.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Framework | Next.js 16 (App Router) | `output: "standalone"` |
| UI library | React 19 | |
| Language | TypeScript (strict mode) | 0 errors required at build time |
| Styling | Tailwind CSS v4, shadcn/ui | |
| LLM SDK | Vercel AI SDK | `generateText`, `withRetry` wrapper |
| Database (web) | Prisma 6 + SQLite | `prisma/dev.db` |
| Database (static) | IndexedDB | For GitHub Pages builds |
| State management | Zustand | Persisted to `localStorage` as `handai-storage` |
| Testing | Vitest | 115 tests across 4 suites |

---

## Project Layout

```
├── src/
│   ├── app/
│   │   ├── api/                    <- Server-side API routes (web deployment only)
│   │   │   ├── process-row/        <- Core LLM dispatch route
│   │   │   ├── consensus-row/      <- Multi-worker + judge logic
│   │   │   ├── local-models/       <- Probes Ollama + LM Studio for available models
│   │   │   └── ...
│   │   ├── ai-coder/               <- AI Coder page
│   │   ├── qualitative-coder/      <- Qualitative Coder page
│   │   ├── consensus-coder/        <- Consensus Coder page
│   │   ├── codebook-generator/     <- Codebook Generator page
│   │   ├── abstract-screener/      <- Abstract Screener page
│   │   ├── transform/              <- Transform page
│   │   ├── automator/              <- Automator page
│   │   ├── generate/               <- Generate page
│   │   ├── process-documents/      <- Process Documents page
│   │   ├── model-comparison/       <- Model Comparison page
│   │   ├── history/                <- History browser + detail page
│   │   └── settings/               <- Settings page
│   ├── components/
│   │   ├── ui/                     <- shadcn/ui primitives (Button, Dialog, etc.)
│   │   ├── tools/                  <- Shared tool components (file upload, column selector, etc.)
│   │   └── AppSidebar.tsx          <- Navigation sidebar + local model detection
│   ├── hooks/                      <- Reusable React hooks (batch processor, restore, etc.)
│   └── lib/
│       ├── ai/
│       │   └── providers.ts        <- getModel() — returns AI SDK model for any provider
│       ├── analytics.ts            <- Cohen's kappa, pairwise agreement calculations
│       ├── db-indexeddb.ts         <- IndexedDB persistence for static builds
│       ├── document-browser.ts     <- Browser-side PDF/DOCX text extraction
│       ├── export.ts               <- downloadCSV() — blob download
│       ├── llm-browser.ts          <- Browser-side LLM functions for static builds
│       ├── llm-dispatch.ts         <- Unified dispatch layer (web vs static)
│       ├── processing-store.ts     <- Global processing state (survives navigation)
│       ├── restore-store.ts        <- Session restore from history
│       ├── prompts.ts              <- Prompt registry + per-tool localStorage overrides
│       ├── retry.ts                <- withRetry() with auth-error fast-fail
│       ├── store.ts                <- Zustand store for provider config
│       └── validation.ts           <- Zod schemas for all API route inputs
├── prisma/
│   ├── schema.prisma               <- Database schema
│   └── dev.db                      <- SQLite database (created on first run)
├── scripts/
│   └── build-static.sh             <- Static export script for GitHub Pages
├── public/                         <- Static assets
├── next.config.ts                  <- Next.js config (build target switched by env var)
├── package.json
└── tsconfig.json
```

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete technical reference, including:

- LLM call path (web vs. static)
- All API routes and their responsibilities
- Database schema (tables, columns, relations)
- Autosave system design
- State management and localStorage keys
- Consensus Coder worker/judge protocol

### LLM call path (quick summary)

**Web app:**
```
Browser -> /api/process-row -> providers.ts (getModel) -> Provider API
                            -> prisma (log result to SQLite)
```

**Static build (GitHub Pages):**
```
Browser -> llm-browser.ts (getModel) -> Provider API (direct, no server)
                                     -> db-indexeddb.ts (log result to IndexedDB)
```

The same React components render in both contexts. The difference is whether the LLM call goes through a Next.js API route (web) or directly from the browser (static).

### Autosave

AI Coder and Abstract Screener write the current session state to `localStorage` after every action:

| Key | Content |
|---|---|
| `aic_autosave` | Current AI Coder session |
| `as_autosave` | Current Abstract Screener session |

On page load, a recovery banner appears if an autosaved session is detected. The user can resume or discard it.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run `npm test` to confirm the baseline passes (115 tests).
3. Make your changes. Add or update tests for any modified behaviour.
4. Run `npm run build` to confirm the TypeScript build passes with 0 errors.
5. Run `npm run lint` and fix any issues — the codebase targets 0 lint errors.
6. Open a pull request with a clear description of what changed and why.

---

## License

See [LICENSE](LICENSE) in the repository root.
