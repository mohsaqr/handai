/**
 * SERVER ONLY — resolves a provider's API key from server-side env vars.
 *
 * Never import this from a client component. Every name below is a plain
 * (non-`NEXT_PUBLIC_`) env var, so Next keeps it out of the browser bundle; an
 * accidental client import would silently read `undefined` rather than leak,
 * but the import itself is still a bug.
 *
 * Pairs with `shared-keys.ts`, which tells the *client* which providers the
 * deployment covers without revealing the key material.
 *
 * Setup for a shared (e.g. classroom) deployment:
 *   OPENAI_API_KEY=sk-...                      # server-only secret
 *   NEXT_PUBLIC_SHARED_KEY_PROVIDERS=openai    # public: which providers to offer
 *   NEXT_PUBLIC_BROWSER_STORAGE=0              # route LLM calls through the server
 */

/** Env var holding each provider's key. */
const ENV_BY_PROVIDER: Record<string, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_GENERATIVE_AI_API_KEY",
  groq: "GROQ_API_KEY",
  together: "TOGETHER_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
  azure: "AZURE_API_KEY",
  custom: "CUSTOM_API_KEY",
};

/** The deployment's key for a provider, or "" when none is configured. */
export function serverApiKey(provider: string): string {
  const envName = ENV_BY_PROVIDER[provider?.toLowerCase()];
  if (!envName) return "";
  return process.env[envName]?.trim() ?? "";
}

/**
 * Pick the key for a request: the caller's own key always wins, so a user with
 * a personal key is never forced onto the shared budget. Falls back to the
 * deployment's key, then to "" (local providers accept an empty key).
 */
export function resolveApiKey(provider: string, apiKey?: string): string {
  return apiKey?.trim() || serverApiKey(provider);
}
