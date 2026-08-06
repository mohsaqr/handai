/**
 * Local (loopback) LLM provider helpers.
 *
 * A "local" provider is one whose endpoint resolves on *whichever machine makes
 * the request*. That makes them fundamentally incompatible with the server-side
 * `/api/*` path: on a hosted deployment, a route handler that dials
 * `http://localhost:1234` reaches the server's own loopback, not the user's
 * machine — which surfaces as `connect ECONNREFUSED 127.0.0.1:1234`.
 *
 * So local providers must always be detected *and* called from the browser,
 * regardless of the deployment's storage mode.
 */

/** Provider ids that are always loopback-backed. */
export const LOCAL_PROVIDER_IDS = new Set(["ollama", "lmstudio"]);

const LOOPBACK_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "::1",
  "[::1]",
]);

/** True when a URL points at the requesting machine (loopback or *.local). */
export function isLoopbackUrl(url?: string): boolean {
  if (!url) return false;
  try {
    const host = new URL(url).hostname.toLowerCase();
    return LOOPBACK_HOSTS.has(host) || host.endsWith(".local");
  } catch {
    return false;
  }
}

/**
 * True when this provider must be reached from the browser.
 * `custom` counts only when its base URL is itself a loopback address.
 */
export function isLocalProvider(provider?: string, baseUrl?: string): boolean {
  if (!provider) return false;
  if (LOCAL_PROVIDER_IDS.has(provider)) return true;
  return provider === "custom" && isLoopbackUrl(baseUrl);
}

/** Strip trailing slashes so URL joins don't double up. */
function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Ollama's model list lives at the server root, not under the /v1 shim. */
export function ollamaTagsUrl(baseUrl?: string): string {
  const fallback = "http://localhost:11434";
  if (!baseUrl) return `${fallback}/api/tags`;
  try {
    return `${new URL(baseUrl).origin}/api/tags`;
  } catch {
    return `${fallback}/api/tags`;
  }
}

/** LM Studio is OpenAI-compatible: GET {baseUrl}/models. */
export function lmStudioModelsUrl(baseUrl?: string): string {
  return `${trimSlash(baseUrl || "http://localhost:1234/v1")}/models`;
}

async function getJson(url: string, timeoutMs: number): Promise<unknown | null> {
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(timeoutMs),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    // Connection refused, CORS, or the browser's local-network gate — all mean
    // "no local models reachable from here".
    return null;
  }
}

export interface LocalBaseUrls {
  ollama?: string;
  lmstudio?: string;
}

/**
 * Probe the local model servers **from the caller's own runtime**.
 *
 * Runs in the browser so that a hosted deployment still detects the models on
 * the user's machine. Honors the base URLs configured in settings rather than
 * assuming the default ports.
 *
 * @returns `{ ollama?: string[], lmstudio?: string[] }` — keys present only for
 *          servers that answered.
 */
export async function probeLocalModels(
  baseUrls: LocalBaseUrls = {},
  timeoutMs = 2000
): Promise<Record<string, string[]>> {
  const [ollama, lm] = await Promise.all([
    getJson(ollamaTagsUrl(baseUrls.ollama), timeoutMs),
    getJson(lmStudioModelsUrl(baseUrls.lmstudio), timeoutMs),
  ]);

  const result: Record<string, string[]> = {};

  const ollamaModels = (ollama as { models?: { name: string }[] } | null)?.models;
  if (ollamaModels) result.ollama = ollamaModels.map((m) => m.name);

  const lmModels = (lm as { data?: { id: string }[] } | null)?.data;
  if (lmModels) result.lmstudio = lmModels.map((m) => m.id);

  return result;
}
