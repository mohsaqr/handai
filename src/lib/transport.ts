/**
 * Transport selection — does an LLM call run in the browser or on the server?
 *
 * Two hard constraints override the deployment's storage mode, in this order:
 *
 * 1. **Local providers must run in the browser.** `localhost` resolves on
 *    whichever machine issues the request, so a server route dialing
 *    `127.0.0.1:1234` reaches the *host's* loopback, not the user's machine.
 * 2. **Shared deployment keys must run on the server.** The key exists only in a
 *    server-side env var; a browser-direct call would send an empty key.
 *
 * Otherwise `useBrowserStorage` decides (browser-direct by default, so API keys
 * never traverse the server).
 */

import { isLocalProvider } from "./local-provider";
import { needsSharedKey } from "./shared-keys";

/** Static web build (GitHub Pages) — no server. Uses IndexedDB + browser-direct LLM. */
export const isStatic = process.env.NEXT_PUBLIC_STATIC === "1";

/**
 * True when the app should operate entirely in the browser:
 * - LLM calls go directly from the browser to provider APIs (no server relay)
 * - Run history is stored in IndexedDB (no server SQLite)
 * - API keys never leave the browser
 *
 * Defaults to true for security — API keys are not sent through the server.
 * Set NEXT_PUBLIC_BROWSER_STORAGE=0 explicitly to use server-side API routes
 * (only for private/self-hosted deployments where the server is trusted).
 */
export const useBrowserStorage =
  process.env.NEXT_PUBLIC_BROWSER_STORAGE === "0" ? false : true;

/** A provider reference carrying enough to decide a transport. */
export type ProviderRef = {
  provider?: string;
  baseUrl?: string;
  apiKey?: string;
};

export type Transport = "browser" | "server";

/**
 * Decide where a call runs. Throws when one call mixes a local provider with a
 * provider that depends on a shared server key — those cannot share a transport,
 * and failing loudly beats a confusing auth or connection error downstream.
 */
export function resolveTransport(refs: ProviderRef | ProviderRef[]): Transport {
  const list = Array.isArray(refs) ? refs : [refs];

  const local = list.filter((r) => isLocalProvider(r.provider, r.baseUrl));
  const shared = list.filter((r) => needsSharedKey(r.provider, r.apiKey));

  if (local.length > 0 && shared.length > 0) {
    const localNames = [...new Set(local.map((r) => r.provider))].join(", ");
    const sharedNames = [...new Set(shared.map((r) => r.provider))].join(", ");
    throw new Error(
      `Cannot mix a local provider (${localNames}) with a provider using this ` +
        `deployment's shared key (${sharedNames}) in one call: local models are ` +
        `reachable only from your browser, while the shared key exists only on ` +
        `the server. Use one kind, or add your own ${sharedNames} key in Settings.`
    );
  }

  if (local.length > 0) return "browser";
  if (shared.length > 0) return "server";
  return useBrowserStorage ? "browser" : "server";
}
