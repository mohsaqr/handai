import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * These modules read env vars at import time, so each case stubs the env and
 * re-imports rather than mutating a cached module.
 */
async function loadShared(list?: string) {
  vi.resetModules();
  if (list === undefined) vi.stubEnv("NEXT_PUBLIC_SHARED_KEY_PROVIDERS", "");
  else vi.stubEnv("NEXT_PUBLIC_SHARED_KEY_PROVIDERS", list);
  return await import("../shared-keys");
}

async function loadTransport(opts: { shared?: string; browserStorage?: string }) {
  vi.resetModules();
  vi.stubEnv("NEXT_PUBLIC_SHARED_KEY_PROVIDERS", opts.shared ?? "");
  vi.stubEnv("NEXT_PUBLIC_BROWSER_STORAGE", opts.browserStorage ?? "");
  return await import("../transport");
}

async function loadServerKeys(env: Record<string, string>) {
  vi.resetModules();
  for (const [k, v] of Object.entries(env)) vi.stubEnv(k, v);
  return await import("../server-keys");
}

beforeEach(() => vi.unstubAllEnvs());
afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("shared-keys", () => {
  it("parses the provider list, tolerating spacing and case", async () => {
    const m = await loadShared(" OpenAI , anthropic ,, ");
    expect(m.hasSharedKey("openai")).toBe(true);
    expect(m.hasSharedKey("anthropic")).toBe(true);
    expect(m.hasSharedKey("groq")).toBe(false);
    expect(m.sharedKeysEnabled()).toBe(true);
  });

  it("is inert when unset", async () => {
    const m = await loadShared();
    expect(m.hasSharedKey("openai")).toBe(false);
    expect(m.sharedKeysEnabled()).toBe(false);
    expect(m.needsSharedKey("openai")).toBe(false);
  });

  it("needsSharedKey yields to a user-supplied key", async () => {
    const m = await loadShared("openai");
    expect(m.needsSharedKey("openai", "")).toBe(true);
    expect(m.needsSharedKey("openai", "   ")).toBe(true);
    expect(m.needsSharedKey("openai", "sk-user-own")).toBe(false);
  });
});

describe("server-keys", () => {
  it("maps providers to their env vars and trims", async () => {
    const m = await loadServerKeys({
      OPENAI_API_KEY: " sk-deployment ",
      ANTHROPIC_API_KEY: "sk-ant",
    });
    expect(m.serverApiKey("openai")).toBe("sk-deployment");
    expect(m.serverApiKey("anthropic")).toBe("sk-ant");
    expect(m.serverApiKey("groq")).toBe("");
    expect(m.serverApiKey("nonsense")).toBe("");
  });

  it("prefers the caller's own key over the deployment key", async () => {
    const m = await loadServerKeys({ OPENAI_API_KEY: "sk-deployment" });
    expect(m.resolveApiKey("openai", "sk-mine")).toBe("sk-mine");
    expect(m.resolveApiKey("openai", "")).toBe("sk-deployment");
    expect(m.resolveApiKey("openai", "  ")).toBe("sk-deployment");
    expect(m.resolveApiKey("openai")).toBe("sk-deployment");
  });

  it("returns empty when neither side has a key", async () => {
    const m = await loadServerKeys({});
    expect(m.resolveApiKey("groq", "")).toBe("");
  });
});

describe("resolveTransport", () => {
  it("defaults to browser-direct", async () => {
    const { resolveTransport } = await loadTransport({});
    expect(resolveTransport({ provider: "openai", apiKey: "sk-x" })).toBe("browser");
  });

  it("honors NEXT_PUBLIC_BROWSER_STORAGE=0", async () => {
    const { resolveTransport } = await loadTransport({ browserStorage: "0" });
    expect(resolveTransport({ provider: "openai", apiKey: "sk-x" })).toBe("server");
  });

  it("forces local providers into the browser even in server mode", async () => {
    const { resolveTransport } = await loadTransport({ browserStorage: "0" });
    expect(resolveTransport({ provider: "lmstudio" })).toBe("browser");
    expect(resolveTransport({ provider: "ollama" })).toBe("browser");
    expect(
      resolveTransport({ provider: "custom", baseUrl: "http://localhost:8080/v1" })
    ).toBe("browser");
  });

  it("forces shared-key providers onto the server even in browser mode", async () => {
    const { resolveTransport } = await loadTransport({ shared: "openai" });
    expect(resolveTransport({ provider: "openai" })).toBe("server");
  });

  it("lets a user's own key keep the default transport", async () => {
    const { resolveTransport } = await loadTransport({ shared: "openai" });
    expect(resolveTransport({ provider: "openai", apiKey: "sk-mine" })).toBe("browser");
  });

  it("routes a whole round to the browser when any participant is local", async () => {
    const { resolveTransport } = await loadTransport({ browserStorage: "0" });
    expect(
      resolveTransport([
        { provider: "openai", apiKey: "sk-x" },
        { provider: "lmstudio" },
      ])
    ).toBe("browser");
  });

  it("throws a clear error when local and shared-key providers are mixed", async () => {
    const { resolveTransport } = await loadTransport({ shared: "openai" });
    expect(() =>
      resolveTransport([{ provider: "lmstudio" }, { provider: "openai" }])
    ).toThrow(/Cannot mix a local provider/);
  });

  it("does not consider a shared provider conflicting when the user has a key", async () => {
    const { resolveTransport } = await loadTransport({ shared: "openai" });
    expect(
      resolveTransport([
        { provider: "lmstudio" },
        { provider: "openai", apiKey: "sk-mine" },
      ])
    ).toBe("browser");
  });
});
