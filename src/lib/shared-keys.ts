/**
 * Shared deployment keys — the "one key for a whole class" setup.
 *
 * A deployment can hold provider API keys in server-only env vars so users never
 * type (or see) a key. This module is the **client-safe half**: it only knows
 * *which* providers the deployment covers, never the key material.
 *
 * Public build-time config:
 *   NEXT_PUBLIC_SHARED_KEY_PROVIDERS=openai,anthropic
 *
 * Server-only secrets live in `server-keys.ts` and must never be imported into
 * client code. The provider list here is deliberately not a secret — it only
 * says which options to offer, which the UI reveals anyway.
 *
 * IMPORTANT: the key is only ever attached inside an `/api/*` route handler, so
 * calls for these providers must take the server transport. See
 * `resolveTransport()` in `transport.ts`.
 */

/** Provider ids this deployment supplies a shared key for. */
export const SHARED_KEY_PROVIDERS: ReadonlySet<string> = new Set(
  (process.env.NEXT_PUBLIC_SHARED_KEY_PROVIDERS ?? "")
    .split(",")
    .map((id) => id.trim().toLowerCase())
    .filter(Boolean)
);

/** True when the deployment holds a key for this provider. */
export function hasSharedKey(providerId?: string): boolean {
  return !!providerId && SHARED_KEY_PROVIDERS.has(providerId.toLowerCase());
}

/** True when any shared key is configured (used to adjust copy in Settings). */
export function sharedKeysEnabled(): boolean {
  return SHARED_KEY_PROVIDERS.size > 0;
}

/**
 * True when this call must be served by the deployment's own key: the provider
 * is covered and the user supplied no key of their own. A user-supplied key
 * always wins, so anyone can still bring their own.
 */
export function needsSharedKey(providerId?: string, apiKey?: string): boolean {
  return !apiKey?.trim() && hasSharedKey(providerId);
}
